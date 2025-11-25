#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Fused Flash Linear Module for DP Training.

This module provides a Linear layer that computes per-sample gradient norms
directly in the backward pass, avoiding the need for hooks and eliminating
FSDP serialization overhead.

The key insight is that for Linear layers, the per-sample gradient norm can be
computed efficiently during the backward pass using the input activations and
output gradients, without materializing the full per-sample gradients.

Supported algorithms:
- 'triton': Fused Triton kernel (fastest, requires CUDA + Triton)
- 'input_length': O(T * d^2) PyTorch implementation
- 'width': O(T^2 * d) PyTorch implementation
- 'auto': Use 'triton' if available on CUDA, else 'input_length'
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import Triton fused kernel (optional dependency)
from opacus.grad_sample.triton_fused_kernel import (
    TRITON_AVAILABLE,
    fused_backward_weight,
    fused_backward_weight_2d,
)


# ============================================================================
# Flash Clipping Kernels
# ============================================================================

@torch.no_grad()
def _input_length_frobenius(
    A: torch.Tensor,  # [B, T, d_a]
    G: torch.Tensor,  # [B, T, d_g]
    dtype_acc=torch.float32,
) -> torch.Tensor:
    """
    Input-Length-Linear Algorithm: O(T * d^2)
    Compute ||A^T @ G||_F^2 per sample efficiently.
    
    This computes the squared Frobenius norm of the per-sample gradient
    for a Linear layer: ||x_i^T @ g_i||_F^2
    
    Args:
        A: Activations tensor [B, T, d_a] or [B, d_a]
        G: Gradient tensor [B, T, d_g] or [B, d_g]
        dtype_acc: Accumulation dtype for numerical stability
        
    Returns:
        Per-sample squared gradient norms [B]
    """
    # Convert to accumulation dtype
    if A.dtype != dtype_acc:
        A = A.to(dtype_acc)
    if G.dtype != dtype_acc:
        G = G.to(dtype_acc)

    # Handle 2D case: [B, d] -> [B, 1, d]
    if A.dim() == 2:
        A = A.unsqueeze(1)
        G = G.unsqueeze(1)

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
    dtype_acc=torch.float32,
) -> torch.Tensor:
    """
    Width-Linear Algorithm: O(T^2 * d) using tiling.
    Optimal when d is very large and T is relatively small.
    
    Args:
        A: Activations tensor [B, T, d_a]
        G: Gradient tensor [B, T, d_g]
        tile_size: Block size for tiling
        dtype_acc: Accumulation dtype for numerical stability
        
    Returns:
        Per-sample squared gradient norms [B]
    """
    # Handle 2D case: [B, d] -> [B, 1, d]
    if A.dim() == 2:
        A = A.unsqueeze(1)
        G = G.unsqueeze(1)
        
    B, T, d_a = A.shape
    _, _, d_g = G.shape
    
    if A.dtype != dtype_acc:
        A = A.to(dtype_acc)
    if G.dtype != dtype_acc:
        G = G.to(dtype_acc)
    
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
    exact per-sample gradient norm contributions in a single backward pass.
    
    This eliminates the need for hooks on Linear layers, avoiding FSDP
    serialization issues during the norm computation pass.
    """
    
    @staticmethod
    def forward(
        ctx, 
        x: torch.Tensor, 
        weight: torch.Tensor, 
        bias: Optional[torch.Tensor], 
        norm_buf: torch.Tensor,
        algorithm: str,
        tile_size: int,
        compute_norms: bool,
    ) -> torch.Tensor:
        """
        Args:
            x: [Batch, ..., In_Dim]
            weight: [Out_Dim, In_Dim]
            bias: [Out_Dim] or None
            norm_buf: [Batch] - buffer to accumulate squared norms
            algorithm: 'input_length' or 'width'
            tile_size: block size for width algorithm
            compute_norms: whether to compute norms in backward
        """
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        ctx.norm_buf = norm_buf
        ctx.algorithm = algorithm
        ctx.tile_size = tile_size
        ctx.compute_norms = compute_norms
        
        # Standard Linear Forward
        output = F.linear(x, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], None, None, None, None]:
        x, weight = ctx.saved_tensors
        norm_buf = ctx.norm_buf
        algorithm = ctx.algorithm
        
        # --- Determine effective algorithm ---
        # 'auto' resolves to 'triton' if available on CUDA, else 'input_length'
        if algorithm == 'auto':
            if TRITON_AVAILABLE and x.is_cuda and x.dim() == 3:
                algorithm = 'triton'
            else:
                algorithm = 'input_length'
        
        # Check if we can use Triton (only for 3D CUDA tensors)
        use_triton = (
            algorithm == 'triton' 
            and TRITON_AVAILABLE 
            and x.is_cuda 
            and x.dim() == 3
            and ctx.compute_norms 
            and norm_buf is not None
        )
        
        # --- 1. Compute grad_x (always standard) ---
        grad_x = grad_out.matmul(weight)
        
        # --- 2. Compute grad_w and norms ---
        if use_triton:
            # FUSED PATH: Triton kernel computes grad_w AND norms in one pass
            # Ensure contiguous for Triton
            x_c = x if x.is_contiguous() else x.contiguous()
            g_c = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
            
            # Triton fused kernel: computes grad_w and accumulates weight norms
            grad_w = fused_backward_weight(x_c, g_c, norm_buf)
            
            # Handle bias gradient and bias norm (cheap, keep separate)
            if ctx.has_bias:
                grad_b = grad_out.sum(dim=(0, 1))
                # Bias norm: ||sum_t(g_{i,t})||^2
                bias_sums = grad_out.sum(dim=1)  # [B, Dout]
                bias_norm_sq = bias_sums.pow(2).sum(dim=1)  # [B]
                norm_buf.add_(bias_norm_sq)
            else:
                grad_b = None
        else:
            # SEPARATE PATH: Standard PyTorch computation
            # Compute grad_w
            if grad_out.dim() == 2:
                grad_w = grad_out.t().matmul(x)
            else:
                grad_w = torch.matmul(
                    grad_out.view(-1, grad_out.shape[-1]).t(), 
                    x.view(-1, x.shape[-1])
                )
            
            grad_b = grad_out.sum(dim=list(range(grad_out.dim() - 1))) if ctx.has_bias else None

            # Compute norms separately
            if ctx.compute_norms and norm_buf is not None:
                if x.dim() == 2:
                    # 2D Case: Use efficient rank-1 formula
                    # ||g_i @ x_i^T||_F^2 = ||g_i||^2 * ||x_i||^2
                    g_sq = grad_out.pow(2).sum(dim=1)
                    x_sq = x.pow(2).sum(dim=1)
                    weight_contrib = g_sq * x_sq
                else:
                    # 3D Case: Use PyTorch flash algorithms
                    if algorithm == 'width':
                        weight_contrib = _width_frobenius(x, grad_out, tile_size=ctx.tile_size)
                    else:
                        # 'input_length' or fallback
                        weight_contrib = _input_length_frobenius(x, grad_out)

                # Add bias norm contribution
                if ctx.has_bias:
                    if grad_out.dim() == 2:
                        bias_contrib = grad_out.pow(2).sum(dim=1)
                    else:
                        sum_over_time = grad_out.sum(dim=1)
                        bias_contrib = sum_over_time.pow(2).sum(dim=1)
                    weight_contrib = weight_contrib + bias_contrib

                # Accumulate into buffer
                if norm_buf.shape[0] != weight_contrib.shape[0]:
                    raise ValueError(f"norm_buf batch {norm_buf.shape[0]} != input batch {weight_contrib.shape[0]}")
                norm_buf.add_(weight_contrib)

        # Return None for non-tensor inputs
        return grad_x, grad_w, grad_b, None, None, None, None


# ============================================================================
# Module Wrapper
# ============================================================================

class FusedFlashLinear(nn.Module):
    """
    A Linear layer that computes per-sample gradient norms directly in the
    backward pass, eliminating the need for hooks.
    
    This is a drop-in replacement for nn.Linear that can be used for
    differential privacy training with FSDP without serialization issues.
    
    Usage:
        1. Replace nn.Linear with FusedFlashLinear
        2. Before forward pass, call set_norm_buffer(norm_buf) with a shared buffer
        3. After backward, the norm_buf contains accumulated squared norms
        
    Algorithm options:
        - 'auto': Use 'triton' if available on CUDA, else 'input_length' (default)
        - 'triton': Fused Triton kernel - fastest, computes grad_w and norms in 1 pass
        - 'input_length': O(T * d^2) PyTorch implementation
        - 'width': O(T^2 * d) PyTorch implementation
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        algorithm: str = "auto",
        tile_size: int = 256,
        device=None,
        dtype=None,
    ):
        """
        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample
            bias: If set to False, the layer will not learn an additive bias
            algorithm: 'auto', 'triton', 'input_length', or 'width'
                       'auto' uses 'triton' if available on CUDA, else 'input_length'
            tile_size: Block size for width algorithm
            device: Device for parameters
            dtype: Dtype for parameters
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.algorithm = algorithm
        self.tile_size = tile_size
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
        self._norm_buf: Optional[torch.Tensor] = None
        self._compute_norms: bool = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def set_norm_buffer(self, norm_buf: Optional[torch.Tensor]):
        """Set the buffer for accumulating per-sample squared norms."""
        self._norm_buf = norm_buf

    def set_compute_norms(self, compute: bool):
        """Enable/disable norm computation in backward pass."""
        self._compute_norms = compute

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # If no norm buffer or norm computation disabled, use standard F.linear
        if self._norm_buf is None or not self._compute_norms:
            return F.linear(input, self.weight, self.bias)
        
        return FusedFlashLinearFn.apply(
            input, 
            self.weight, 
            self.bias, 
            self._norm_buf,
            self.algorithm,
            self.tile_size,
            self._compute_norms,
        )
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, algorithm={self.algorithm}'


def replace_linear_with_fused(
    module: nn.Module,
    algorithm: str = "auto",
    tile_size: int = 256,
) -> nn.Module:
    """
    Recursively replace all nn.Linear modules with FusedFlashLinear.
    
    Args:
        module: The module to process
        algorithm: Algorithm for norm computation
                   'auto' (default), 'triton', 'input_length', or 'width'
        tile_size: Tile size for width algorithm
        
    Returns:
        The modified module with Linear layers replaced.
        If the module itself is nn.Linear, returns a new FusedFlashLinear.
    """
    # Handle case where module itself is nn.Linear
    if isinstance(module, nn.Linear) and not isinstance(module, FusedFlashLinear):
        fused = FusedFlashLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            algorithm=algorithm,
            tile_size=tile_size,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        fused.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            fused.bias.data.copy_(module.bias.data)
        return fused
    
    # Recurse into children
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and not isinstance(child, FusedFlashLinear):
            # Create FusedFlashLinear with same configuration
            fused = FusedFlashLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                algorithm=algorithm,
                tile_size=tile_size,
                device=child.weight.device,
                dtype=child.weight.dtype,
            )
            # Copy weights
            fused.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                fused.bias.data.copy_(child.bias.data)
            # Replace the module
            setattr(module, name, fused)
        else:
            # Recurse into children
            replace_linear_with_fused(child, algorithm, tile_size)
    
    return module


def get_fused_linear_modules(module: nn.Module) -> list:
    """
    Get all FusedFlashLinear modules in the model.
    
    Args:
        module: The module to search
        
    Returns:
        List of FusedFlashLinear modules
    """
    fused_modules = []
    for m in module.modules():
        if isinstance(m, FusedFlashLinear):
            fused_modules.append(m)
    return fused_modules


__all__ = [
    "TRITON_AVAILABLE",
    "FusedFlashLinear",
    "FusedFlashLinearFn",
    "replace_linear_with_fused",
    "get_fused_linear_modules",
]

