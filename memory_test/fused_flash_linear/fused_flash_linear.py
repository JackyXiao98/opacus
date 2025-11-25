import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from typing import Optional, Tuple

# ============================================================================
# Flash Clipping Kernels (Ported from provided implementation)
# ============================================================================

@torch.no_grad()
def _input_length_frobenius(
    A: torch.Tensor,  # [B, T, d_a]
    G: torch.Tensor,  # [B, T, d_g]
    dtype_acc = torch.float32,
) -> torch.Tensor:
    """
    Input-Length-Linear Algorithm: O(T * d^2)
    Compute ||A^T @ G||_F^2 per sample efficiently.
    """
    # Convert to accumulation dtype
    if A.dtype != dtype_acc: A = A.to(dtype_acc)
    if G.dtype != dtype_acc: G = G.to(dtype_acc)

    # Step 1: Transpose A: [B, d_a, T]
    A_t = A.transpose(1, 2)
    
    # Step 2: Batch matrix multiply: [B, d_a, T] @ [B, T, d_g] -> [B, d_a, d_g]
    # This matrix S represents the per-sample gradient dL/dW
    S = torch.bmm(A_t, G)
    
    # Step 3: Element-wise square and sum to get Frobenius norm squared
    return torch.sum(S * S, dim=(1, 2))  # [B]

@torch.no_grad()
def _width_frobenius(
    A: torch.Tensor,  # [B, T, d_a]
    G: torch.Tensor,  # [B, T, d_g]
    tile_size: int = 256,
    dtype_acc = torch.float32,
) -> torch.Tensor:
    """
    Width-Linear Algorithm: O(T^2 * d) using tiling.
    Optimal when d is very large and T is relatively small.
    """
    B, T, d_a = A.shape
    _, _, d_g = G.shape
    
    if A.dtype != dtype_acc: A = A.to(dtype_acc)
    if G.dtype != dtype_acc: G = G.to(dtype_acc)
    
    total_norm_squared = torch.zeros(B, dtype=dtype_acc, device=A.device)
    num_tiles = (T + tile_size - 1) // tile_size
    
    for j in range(num_tiles):
        j_start = j * tile_size
        j_end = min((j + 1) * tile_size, T)
        
        a_j = A[:, j_start:j_end, :]
        g_j = G[:, j_start:j_end, :]
        
        for k in range(j, num_tiles):
            k_start = k * tile_size
            k_end = min((k + 1) * tile_size, T)
            
            a_k = A[:, k_start:k_end, :]
            g_k = G[:, k_start:k_end, :]
            
            # Score_a = a_j @ a_k^T: [B, tau_j, tau_k]
            Score_a = torch.bmm(a_j, a_k.transpose(1, 2))
            # Score_g = g_j @ g_k^T: [B, tau_j, tau_k]
            Score_g = torch.bmm(g_j, g_k.transpose(1, 2))
            
            block_sum = torch.sum(Score_a * Score_g, dim=(1, 2))
            
            if j == k:
                total_norm_squared.add_(block_sum)
            else:
                total_norm_squared.add_(block_sum, alpha=2.0)
    
    return total_norm_squared

# ============================================================================
# Fused Autograd Function
# ============================================================================

class FusedFlashLinearFn(torch.autograd.Function):
    """
    Fused autograd function that computes standard gradients AND accumulates
    Exact Flash Clipping norm contributions in a single backward pass.
    """
    
    @staticmethod
    def forward(
        ctx, 
        x: torch.Tensor, 
        weight: torch.Tensor, 
        bias: Optional[torch.Tensor], 
        norm_buf: torch.Tensor,
        algorithm: str,
        tile_size: int
    ) -> torch.Tensor:
        """
        Args:
            x: [Batch, ..., In_Dim]
            weight: [Out_Dim, In_Dim]
            bias: [Out_Dim]
            norm_buf: [Batch]
            algorithm: 'input_length' or 'width'
            tile_size: block size for width algorithm
        """
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        ctx.norm_buf = norm_buf
        ctx.algorithm = algorithm
        ctx.tile_size = tile_size
        
        # Standard Linear Forward
        output = F.linear(x, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        x, weight = ctx.saved_tensors
        norm_buf = ctx.norm_buf
        
        # --- 1. Compute Standard Gradients ---
        # grad_x: [B, ..., In]
        grad_x = grad_out.matmul(weight)
        
        # grad_w: [Out, In]
        if grad_out.dim() == 2:
            grad_w = grad_out.t().matmul(x)
        else:
            # Flatten batch and seq for standard weight grad
            grad_w = torch.matmul(
                grad_out.view(-1, grad_out.shape[-1]).t(), 
                x.view(-1, x.shape[-1])
            )
            
        grad_b = grad_out.sum(dim=list(range(grad_out.dim() - 1))) if ctx.has_bias else None

        # --- 2. Compute Flash Norm Contribution (In-Place) ---
        # Unlike Ghost Clipping (Upper Bound), Flash Clipping computes the EXACT
        # per-sample gradient norm.
        
        # Calculate contribution for Weight Gradients
        if x.dim() == 2:
            # 2D Case [B, D]: Exact Norm == Ghost Norm (Outer product rank-1)
            # ||g_i * x_i^T||^2 = ||g_i||^2 * ||x_i||^2
            g_sq = grad_out.pow(2).sum(dim=1)
            x_sq = x.pow(2).sum(dim=1)
            weight_contrib = g_sq * x_sq
        else:
            # 3D Case [B, T, D]: Use Flash Algorithms
            # Assume Batch is dim 0
            if ctx.algorithm == 'width':
                weight_contrib = _width_frobenius(x, grad_out, tile_size=ctx.tile_size)
            else:
                # Default to input_length (usually faster for standard seq lengths)
                weight_contrib = _input_length_frobenius(x, grad_out)

        # Calculate contribution for Bias Gradients
        if ctx.has_bias:
            # Bias grad per sample i is sum_t(grad_out[i, t, :])
            # We need ||sum_t(g_{i,t})||^2
            if grad_out.dim() == 2:
                # 2D case: grad_out is already the per-sample grad
                bias_contrib = grad_out.pow(2).sum(dim=1)
            else:
                # 3D case: Sum over time (dim 1) then norm squared
                # sum_over_time: [B, D_out]
                sum_over_time = grad_out.sum(dim=1)
                bias_contrib = sum_over_time.pow(2).sum(dim=1)
            
            # Total contribution = Weight Norm + Bias Norm
            # (Note: Squared norms are additive)
            weight_contrib.add_(bias_contrib)

        # Accumulate into global buffer
        if norm_buf is not None:
            if norm_buf.shape[0] != weight_contrib.shape[0]:
                raise ValueError(f"norm_buf batch {norm_buf.shape[0]} != input batch {weight_contrib.shape[0]}")
            norm_buf.add_(weight_contrib)

        # Return None for non-tensor inputs (norm_buf, algorithm, tile_size)
        return grad_x, grad_w, grad_b, None, None, None

# ============================================================================
# Module Wrapper
# ============================================================================

class FusedFlashLinear(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        algorithm: str = "input_length",
        tile_size: int = 256
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.algorithm = algorithm
        self.tile_size = tile_size
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
        self._norm_buf: Optional[torch.Tensor] = None

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def set_norm_buffer(self, norm_buf: torch.Tensor):
        self._norm_buf = norm_buf

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._norm_buf is None:
            return F.linear(input, self.weight, self.bias)
        
        return FusedFlashLinearFn.apply(
            input, 
            self.weight, 
            self.bias, 
            self._norm_buf,
            self.algorithm,
            self.tile_size
        )

import math

# --- Usage Example ---

def example_usage():
    # Setup
    B, S, D_in, D_out = 4, 32, 64, 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize with 'input_length' algorithm (O(T*d^2))
    model = FusedFlashLinear(D_in, D_out, bias=True, algorithm="input_length").to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Mock Data (3D Input for Flash logic)
    x = torch.randn(B, S, D_in, device=device)
    y = torch.randn(B, S, D_out, device=device)
    
    optimizer.zero_grad()
    
    # 1. Init Buffer
    norm_buf = torch.zeros(B, device=device)
    model.set_norm_buffer(norm_buf)
    
    # 2. Forward
    pred = model(x)
    loss = ((pred - y) ** 2).mean()
    
    # 3. Backward (Computes exact norms internally)
    loss.backward()
    
    print(f"Norm buf (Exact per-sample norm squared): {norm_buf.data}")
    
    # 4. DP Clip & Step
    if dist.is_initialized():
        dist.all_reduce(norm_buf, op=dist.ReduceOp.SUM)
        
    per_sample_norms = torch.sqrt(norm_buf + 1e-6)
    clip_factor = (1.0 / per_sample_norms).clamp(max=1.0)
    global_scale = clip_factor.mean().item()
    
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(global_scale)
            
    optimizer.step()
    print("Step complete.")

if __name__ == "__main__":
    example_usage()