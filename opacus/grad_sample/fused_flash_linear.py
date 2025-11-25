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

# Try to import DTensor for FSDP support
try:
    from torch.distributed._tensor import DTensor
    DTENSOR_AVAILABLE = True
except ImportError:
    DTensor = None
    DTENSOR_AVAILABLE = False


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
    
    Supports two modes:
    - Normal mode: Computes grad_w, grad_b, and norms in backward
    - Bookkeeping mode: Computes norms only, caches x and grad_out for later
                        clipped gradient computation
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
        compute_norms_container: dict,
        enable_bookkeeping_container: dict,
        module_ref,
    ) -> torch.Tensor:
        """
        Args:
            x: [Batch, ..., In_Dim]
            weight: [Out_Dim, In_Dim]
            bias: [Out_Dim] or None
            norm_buf: [Batch] - buffer to accumulate squared norms
            algorithm: 'input_length' or 'width'
            tile_size: block size for width algorithm
            compute_norms_container: mutable dict with 'value' key for live compute_norms state
                                     Using a container allows backward to see updated state
                                     (e.g., after disable_hooks() is called between two backward passes)
            enable_bookkeeping_container: mutable dict for bookkeeping mode flag
            module_ref: reference to FusedFlashLinear module for caching in BK mode
        """
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        ctx.norm_buf = norm_buf
        ctx.algorithm = algorithm
        ctx.tile_size = tile_size
        # Store the container reference, not the value - this allows backward
        # to see the current state even if it changes after forward
        ctx.compute_norms_container = compute_norms_container
        ctx.enable_bookkeeping_container = enable_bookkeeping_container
        ctx.module_ref = module_ref
        
        # Standard Linear Forward
        output = F.linear(x, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x, weight = ctx.saved_tensors
        norm_buf = ctx.norm_buf
        algorithm = ctx.algorithm
        # Read current states from containers (may have changed since forward)
        compute_norms = ctx.compute_norms_container['value']
        enable_bookkeeping = ctx.enable_bookkeeping_container['value']
        
        # --- Determine effective algorithm ---
        # 'auto' resolves to 'triton' if available on CUDA, else 'input_length'
        if algorithm == 'auto':
            if TRITON_AVAILABLE and x.is_cuda and x.dim() == 3:
                algorithm = 'triton'
            else:
                algorithm = 'input_length'
        
        # --- 1. Compute grad_x (always needed for gradient flow) ---
        grad_x = grad_out.matmul(weight)
        
        # --- 2. Bookkeeping mode: cache and compute norms only ---
        if enable_bookkeeping:
            # Cache x and grad_out for later clipped gradient computation
            ctx.module_ref._bk_cache = {
                'x': x.detach(),
                'grad_out': grad_out.detach(),
            }
            
            # Compute norms (still needed for clipping coefficient)
            # Use Triton if available for 3D CUDA tensors
            use_triton_for_norms = (
                TRITON_AVAILABLE
                and x.is_cuda
                and x.dim() == 3
            )
            
            if compute_norms and norm_buf is not None:
                if use_triton_for_norms:
                    # Use Triton fused kernel for norm computation (discard grad_w)
                    x_c = x if x.is_contiguous() else x.contiguous()
                    g_c = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
                    # fused_backward_weight computes grad_w AND accumulates norms
                    # We discard grad_w since we'll compute clipped gradients later
                    _ = fused_backward_weight(x_c, g_c, norm_buf)
                    
                    # Add bias norm contribution
                    if ctx.has_bias:
                        bias_sums = grad_out.sum(dim=1)  # [B, Dout]
                        bias_norm_sq = bias_sums.pow(2).sum(dim=1)  # [B]
                        norm_buf.add_(bias_norm_sq)
                elif x.dim() == 2:
                    # 2D Case: ||g_i @ x_i^T||_F^2 = ||g_i||^2 * ||x_i||^2
                    g_sq = grad_out.pow(2).sum(dim=1)
                    x_sq = x.pow(2).sum(dim=1)
                    weight_contrib = g_sq * x_sq
                    
                    # Add bias norm contribution
                    if ctx.has_bias:
                        bias_contrib = grad_out.pow(2).sum(dim=1)
                        weight_contrib = weight_contrib + bias_contrib
                    
                    norm_buf.add_(weight_contrib)
                else:
                    # 3D Case: PyTorch fallback
                    if algorithm == 'width':
                        weight_contrib = _width_frobenius(x, grad_out, tile_size=ctx.tile_size)
                    else:
                        weight_contrib = _input_length_frobenius(x, grad_out)
                    
                    # Add bias norm contribution
                    if ctx.has_bias:
                        sum_over_time = grad_out.sum(dim=1)
                        bias_contrib = sum_over_time.pow(2).sum(dim=1)
                        weight_contrib = weight_contrib + bias_contrib
                    
                    norm_buf.add_(weight_contrib)
            
            # Return None for grad_w and grad_b - we'll compute clipped gradients later
            return grad_x, None, None, None, None, None, None, None, None
        
        # --- 3. Normal mode: Compute grad_w and norms ---
        # Check if we can use Triton (only for 3D CUDA tensors)
        use_triton = (
            TRITON_AVAILABLE 
            and x.is_cuda 
            and x.dim() == 3
            and compute_norms 
            and norm_buf is not None
        )
        
        if use_triton:
            # FUSED PATH: Triton kernel computes grad_w AND norms in one pass
            x_c = x if x.is_contiguous() else x.contiguous()
            g_c = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
            
            grad_w = fused_backward_weight(x_c, g_c, norm_buf)
            
            if ctx.has_bias:
                grad_b = grad_out.sum(dim=(0, 1))
                bias_sums = grad_out.sum(dim=1)
                bias_norm_sq = bias_sums.pow(2).sum(dim=1)
                norm_buf.add_(bias_norm_sq)
            else:
                grad_b = None
        else:
            # SEPARATE PATH: Standard PyTorch computation
            if grad_out.dim() == 2:
                grad_w = grad_out.t().matmul(x)
            else:
                grad_w = torch.matmul(
                    grad_out.view(-1, grad_out.shape[-1]).t(), 
                    x.view(-1, x.shape[-1])
                )
            
            grad_b = grad_out.sum(dim=list(range(grad_out.dim() - 1))) if ctx.has_bias else None

            # Compute norms separately
            if compute_norms and norm_buf is not None:
                if x.dim() == 2:
                    g_sq = grad_out.pow(2).sum(dim=1)
                    x_sq = x.pow(2).sum(dim=1)
                    weight_contrib = g_sq * x_sq
                else:
                    if algorithm == 'width':
                        weight_contrib = _width_frobenius(x, grad_out, tile_size=ctx.tile_size)
                    else:
                        weight_contrib = _input_length_frobenius(x, grad_out)

                if ctx.has_bias:
                    if grad_out.dim() == 2:
                        bias_contrib = grad_out.pow(2).sum(dim=1)
                    else:
                        sum_over_time = grad_out.sum(dim=1)
                        bias_contrib = sum_over_time.pow(2).sum(dim=1)
                    weight_contrib = weight_contrib + bias_contrib

                if norm_buf.shape[0] != weight_contrib.shape[0]:
                    raise ValueError(f"norm_buf batch {norm_buf.shape[0]} != input batch {weight_contrib.shape[0]}")
                norm_buf.add_(weight_contrib)

        # Return None for non-tensor inputs (9 total: norm_buf, algorithm, tile_size, 
        # compute_norms_container, enable_bookkeeping_container, module_ref)
        return grad_x, grad_w, grad_b, None, None, None, None, None, None


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
        # Use a mutable container for compute_norms flag so that changes
        # after forward() are visible in backward() (important for two-pass ghost clipping)
        self._compute_norms_container: dict = {'value': False}
        
        # Bookkeeping mode: cache activations/backprops for single-pass clipping
        # Uses mutable container so backward can see current state
        self._enable_bookkeeping_container: dict = {'value': False}
        self._bk_cache: Optional[dict] = None  # {'x': tensor, 'grad_out': tensor}

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
        """
        Enable/disable norm computation in backward pass.
        
        Uses a mutable container so that changes after forward() but before
        backward() are visible. This is critical for two-pass ghost clipping:
        - First backward: compute_norms=True → compute per-sample norms
        - disable_hooks() → compute_norms=False
        - Second backward: compute_norms=False → skip redundant norm computation
        """
        self._compute_norms_container['value'] = compute

    def set_bookkeeping_mode(self, enable: bool):
        """
        Enable/disable bookkeeping mode for single-pass clipping.
        
        In bookkeeping mode:
        - backward() caches x and grad_out instead of computing grad_w
        - compute_clipped_gradient() is called later to compute clipped gradients
        
        This avoids computing gradients twice (once in backward, once with clipping).
        """
        self._enable_bookkeeping_container['value'] = enable

    def compute_clipped_gradient(self, clipping_coef: torch.Tensor):
        """
        Compute clipped gradients from cached activations and backprops.
        
        Uses the mathematical property:
        clipped_grad_w = sum_i(c_i * g_i^T @ x_i) = (c * g)^T @ x
        
        This computes exact per-sample clipped gradients without materializing
        per-sample gradient matrices.
        
        Args:
            clipping_coef: Per-sample clipping coefficients [batch_size]
        """
        if self._bk_cache is None:
            raise RuntimeError(
                "No cached data for clipped gradient computation. "
                "Make sure bookkeeping mode is enabled and backward() was called."
            )
        
        x = self._bk_cache['x']
        grad_out = self._bk_cache['grad_out']
        
        # Scale grad_out by per-sample clipping coefficients
        # x: [B, T, Din] or [B, Din], grad_out: [B, T, Dout] or [B, Dout]
        if grad_out.dim() == 2:
            # 2D case: [B, Dout]
            scaled_g = grad_out * clipping_coef.view(-1, 1).to(device=grad_out.device, dtype=grad_out.dtype)
            # grad_w = scaled_g.T @ x -> [Dout, B] @ [B, Din] -> [Dout, Din]
            grad_w = scaled_g.t().matmul(x)
            # grad_b = scaled_g.sum(dim=0) -> [Dout]
            grad_b = scaled_g.sum(dim=0) if self.bias is not None else None
        else:
            # 3D case: [B, T, Dout]
            scaled_g = grad_out * clipping_coef.view(-1, 1, 1).to(device=grad_out.device, dtype=grad_out.dtype)
            # Reshape for matmul: [B*T, Dout].T @ [B*T, Din] -> [Dout, Din]
            grad_w = torch.matmul(
                scaled_g.view(-1, scaled_g.shape[-1]).t(),
                x.view(-1, x.shape[-1])
            )
            # grad_b = scaled_g.sum(dim=(0, 1)) -> [Dout]
            grad_b = scaled_g.sum(dim=(0, 1)) if self.bias is not None else None
        
        # Set gradients on parameters (ensure dtype matches parameter dtype)
        # Handle DTensor case for FSDP compatibility
        grad_w = grad_w.to(dtype=self.weight.dtype)
        
        # Check if weight is a DTensor (FSDP sharded)
        if DTENSOR_AVAILABLE and isinstance(self.weight, DTensor):
            # Convert gradient to DTensor with same spec as parameter
            # The gradient is computed on the full (replicated) tensor, so we use Replicate placement
            from torch.distributed._tensor import Replicate
            grad_w = DTensor.from_local(
                grad_w, 
                device_mesh=self.weight.device_mesh,
                placements=[Replicate()],
            )
        
        if self.weight.grad is None:
            self.weight.grad = grad_w
        else:
            self.weight.grad.add_(grad_w)
        
        if self.bias is not None and grad_b is not None:
            grad_b = grad_b.to(dtype=self.bias.dtype)
            
            # Check if bias is a DTensor (FSDP sharded)
            if DTENSOR_AVAILABLE and isinstance(self.bias, DTensor):
                from torch.distributed._tensor import Replicate
                grad_b = DTensor.from_local(
                    grad_b,
                    device_mesh=self.bias.device_mesh,
                    placements=[Replicate()],
                )
            
            if self.bias.grad is None:
                self.bias.grad = grad_b
            else:
                self.bias.grad.add_(grad_b)

    def clear_bk_cache(self):
        """Clear bookkeeping cache to free memory."""
        if self._bk_cache is not None:
            self._bk_cache.clear()
            self._bk_cache = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # If no norm buffer or norm computation disabled, use standard F.linear
        if self._norm_buf is None or not self._compute_norms_container['value']:
            return F.linear(input, self.weight, self.bias)
        
        return FusedFlashLinearFn.apply(
            input, 
            self.weight, 
            self.bias, 
            self._norm_buf,
            self.algorithm,
            self.tile_size,
            self._compute_norms_container,
            self._enable_bookkeeping_container,
            self,  # Pass module reference for caching in BK mode
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

