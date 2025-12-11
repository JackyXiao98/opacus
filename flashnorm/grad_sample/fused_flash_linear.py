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

Uses Triton fused kernel for optimal performance on CUDA GPUs.
Supports inputs of any dimension >= 2 by reshaping to [B, T, D] format.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import Triton fused kernel (optional dependency)
from flashnorm.grad_sample.triton_fused_kernel import (
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
# Utility Functions
# ============================================================================

def _reshape_to_3d(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    Reshape tensor to 3D [B, T, D] format for Triton kernel.
    
    Args:
        x: Input tensor of shape [B, ...dims..., D]
        
    Returns:
        Tuple of (reshaped tensor [B, T, D], original middle dims)
        where T = product of middle dimensions
    """
    if x.dim() == 2:
        # [B, D] -> [B, 1, D]
        return x.unsqueeze(1), (1,)
    elif x.dim() == 3:
        # Already 3D
        return x, (x.shape[1],)
    else:
        # [B, d1, d2, ..., D] -> [B, d1*d2*..., D]
        B = x.shape[0]
        D = x.shape[-1]
        middle_dims = x.shape[1:-1]
        T = 1
        for d in middle_dims:
            T *= d
        return x.view(B, T, D), middle_dims


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
    
    Handles inputs of any dimension >= 2 by reshaping to [B, T, D] format.
    """
    
    @staticmethod
    def forward(
        ctx, 
        x: torch.Tensor, 
        weight: torch.Tensor, 
        bias: Optional[torch.Tensor], 
        norm_buf: torch.Tensor,
        compute_norms_container: dict,
        enable_bookkeeping_container: dict,
        module_ref,
    ) -> torch.Tensor:
        """
        Args:
            x: [Batch, ..., In_Dim] - supports 2D, 3D, 4D+ inputs
            weight: [Out_Dim, In_Dim]
            bias: [Out_Dim] or None
            norm_buf: [Batch] - buffer to accumulate squared norms
            compute_norms_container: mutable dict with 'value' key for live compute_norms state
            enable_bookkeeping_container: mutable dict for bookkeeping mode flag
            module_ref: reference to FusedFlashLinear module for caching in BK mode
        """
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        ctx.norm_buf = norm_buf
        ctx.original_shape = x.shape
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
        original_shape = ctx.original_shape
        # Read current states from containers (may have changed since forward)
        compute_norms = ctx.compute_norms_container['value']
        enable_bookkeeping = ctx.enable_bookkeeping_container['value']
        
        # --- 1. Compute grad_x (always needed for gradient flow) ---
        grad_x = grad_out.matmul(weight)
        
        # --- 2. Reshape to 3D for Triton kernel ---
        # Handle 2D, 3D, and 4D+ cases uniformly
        original_dim = x.dim()
        if original_dim == 2:
            x_3d = x.unsqueeze(1)  # [B, D] -> [B, 1, D]
            g_3d = grad_out.unsqueeze(1)  # [B, D] -> [B, 1, D]
        elif original_dim == 3:
            x_3d = x
            g_3d = grad_out
        else:
            # 4D+ case: [B, d1, d2, ..., D] -> [B, T, D] where T = d1*d2*...
            B = x.shape[0]
            Din = x.shape[-1]
            Dout = grad_out.shape[-1]
            x_3d = x.view(B, -1, Din)
            g_3d = grad_out.view(B, -1, Dout)
        
        # --- 3. Bookkeeping mode: cache and compute norms only ---
        if enable_bookkeeping:
            # Cache x and grad_out for later clipped gradient computation
            ctx.module_ref._bk_cache = {
                'x': x.detach(),
                'grad_out': grad_out.detach(),
            }
            
            if compute_norms and norm_buf is not None:
                # Use Triton fused kernel for norm computation (discard grad_w)
                if TRITON_AVAILABLE and x.is_cuda:
                    x_c = x_3d if x_3d.is_contiguous() else x_3d.contiguous()
                    g_c = g_3d if g_3d.is_contiguous() else g_3d.contiguous()
                    _ = fused_backward_weight(x_c, g_c, norm_buf)
                    
                    # Add bias norm contribution
                    if ctx.has_bias:
                        # Sum over all dims except batch and last
                        if original_dim == 2:
                            bias_norm_sq = grad_out.pow(2).sum(dim=1)
                        else:
                            bias_sums = grad_out.sum(dim=tuple(range(1, original_dim - 1)))  # [B, Dout]
                            bias_norm_sq = bias_sums.pow(2).sum(dim=1)  # [B]
                        norm_buf.add_(bias_norm_sq)
                else:
                    # CPU fallback: use 2D efficient formula
                    if original_dim == 2:
                        g_sq = grad_out.pow(2).sum(dim=1)
                        x_sq = x.pow(2).sum(dim=1)
                        weight_contrib = g_sq * x_sq
                        
                        if ctx.has_bias:
                            weight_contrib = weight_contrib + grad_out.pow(2).sum(dim=1)
                        norm_buf.add_(weight_contrib)
                    else:
                        # Use fused_backward_weight_2d equivalent logic for 3D+
                        # Materialize per-sample gradients for norm computation
                        # grad_w_i = g_i^T @ x_i, norm = ||grad_w_i||_F^2
                        x_c = x_3d if x_3d.is_contiguous() else x_3d.contiguous()
                        g_c = g_3d if g_3d.is_contiguous() else g_3d.contiguous()
                        # [B, T, Din].transpose(1,2) @ [B, T, Dout] -> [B, Din, Dout]
                        per_sample_grad = torch.bmm(x_c.transpose(1, 2), g_c)
                        weight_contrib = per_sample_grad.pow(2).sum(dim=(1, 2))
                        
                        if ctx.has_bias:
                            bias_sums = g_c.sum(dim=1)  # [B, Dout]
                            weight_contrib = weight_contrib + bias_sums.pow(2).sum(dim=1)
                        norm_buf.add_(weight_contrib)
            
            # Return None for grad_w and grad_b - we'll compute clipped gradients later
            return grad_x, None, None, None, None, None, None
        
        # --- 4. Normal mode: Compute grad_w and norms ---
        use_triton = TRITON_AVAILABLE and x.is_cuda and compute_norms and norm_buf is not None
        
        if use_triton:
            # FUSED PATH: Triton kernel computes grad_w AND norms in one pass
            x_c = x_3d if x_3d.is_contiguous() else x_3d.contiguous()
            g_c = g_3d if g_3d.is_contiguous() else g_3d.contiguous()
            
            grad_w = fused_backward_weight(x_c, g_c, norm_buf)
            
            if ctx.has_bias:
                # Sum over all dims except last
                # Use g_c (not g_3d) to ensure contiguous memory access
                # g_c is guaranteed to be contiguous and was used in the Triton kernel
                if original_dim == 2:
                    grad_b = grad_out.sum(dim=0)
                else:
                    # For 3D+ inputs, use g_c which is contiguous [B, T, Dout]
                    # Sum over batch and sequence dimensions: [B, T, Dout] -> [Dout]
                    grad_b = g_c.sum(dim=(0, 1))
                
                # Bias norm: sum over sequence dims first, then compute per-sample norm
                if original_dim == 2:
                    bias_norm_sq = grad_out.pow(2).sum(dim=1)
                else:
                    # Use g_c for consistency and to avoid memory access issues
                    bias_sums = g_c.sum(dim=1)  # [B, Dout]
                    bias_norm_sq = bias_sums.pow(2).sum(dim=1)  # [B]
                norm_buf.add_(bias_norm_sq)
            else:
                grad_b = None
        else:
            # STANDARD PATH: PyTorch computation
            # grad_w = g^T @ x (reshape appropriately)
            grad_w = torch.matmul(
                g_3d.view(-1, g_3d.shape[-1]).t(),
                x_3d.view(-1, x_3d.shape[-1])
            )
            
            # Use g_3d for consistency and to avoid memory access issues with 4D+ inputs
            if ctx.has_bias:
                if original_dim == 2:
                    grad_b = grad_out.sum(dim=0)
                else:
                    # For 3D+ inputs, use g_3d which is already reshaped to [B, T, Dout]
                    grad_b = g_3d.sum(dim=(0, 1))
            else:
                grad_b = None
            
            # Compute norms if needed
            if compute_norms and norm_buf is not None:
                if original_dim == 2:
                    # 2D efficient: ||g_i @ x_i^T||_F^2 = ||g_i||^2 * ||x_i||^2
                    g_sq = grad_out.pow(2).sum(dim=1)
                    x_sq = x.pow(2).sum(dim=1)
                    weight_contrib = g_sq * x_sq
                else:
                    # Materialize per-sample gradients
                    x_c = x_3d if x_3d.is_contiguous() else x_3d.contiguous()
                    g_c = g_3d if g_3d.is_contiguous() else g_3d.contiguous()
                    per_sample_grad = torch.bmm(x_c.transpose(1, 2), g_c)
                    weight_contrib = per_sample_grad.pow(2).sum(dim=(1, 2))
                
                if ctx.has_bias:
                    if original_dim == 2:
                        bias_contrib = grad_out.pow(2).sum(dim=1)
                    else:
                        bias_sums = g_3d.sum(dim=1)
                        bias_contrib = bias_sums.pow(2).sum(dim=1)
                    weight_contrib = weight_contrib + bias_contrib
                
                norm_buf.add_(weight_contrib)

        # Return gradients (7 inputs: x, weight, bias, norm_buf, 3 containers)
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
    
    Uses Triton fused kernel for optimal performance. Supports inputs of
    any dimension >= 2 (2D: [B, D], 3D: [B, T, D], 4D+: [B, d1, d2, ..., D]).
    
    Usage:
        1. Replace nn.Linear with FusedFlashLinear
        2. Before forward pass, call set_norm_buffer(norm_buf) with a shared buffer
        3. After backward, the norm_buf contains accumulated squared norms
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        """
        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample
            bias: If set to False, the layer will not learn an additive bias
            device: Device for parameters
            dtype: Dtype for parameters
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
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

    def compute_clipped_gradient(self, clipping_coef: torch.Tensor, grad_scale: float = 1.0):
        """
        Compute clipped gradients from cached activations and backprops.
        
        Uses the mathematical property:
        clipped_grad_w = sum_i(c_i * g_i^T @ x_i) = (c * g)^T @ x
        
        This computes exact per-sample clipped gradients without materializing
        per-sample gradient matrices.
        
        Handles inputs of any dimension >= 2.
        
        Args:
            clipping_coef: Per-sample clipping coefficients [batch_size]
            grad_scale: Scale factor for gradients (e.g., batch_size for loss_reduction="mean")
                       When loss_reduction="mean", grad_out is divided by batch_size,
                       so we need to multiply by batch_size to get correct gradient magnitude.
        """
        if self._bk_cache is None:
            raise RuntimeError(
                "No cached data for clipped gradient computation. "
                "Make sure bookkeeping mode is enabled and backward() was called."
            )
        
        x = self._bk_cache['x']
        grad_out = self._bk_cache['grad_out']
        original_dim = x.dim()
        
        # Scale grad_out by per-sample clipping coefficients and grad_scale
        # Use in-place operations to reduce memory allocation
        # Safe to modify grad_out since cache will be cleared after this computation
        coef_with_scale = clipping_coef * grad_scale
        
        if original_dim == 2:
            # 2D case: [B, Din], [B, Dout]
            # Use in-place multiplication to avoid creating a new tensor
            coef_view = coef_with_scale.view(-1, 1).to(device=grad_out.device, dtype=grad_out.dtype)
            grad_out.mul_(coef_view)
            grad_w = grad_out.t().matmul(x)
            grad_b = grad_out.sum(dim=0) if self.bias is not None else None
        else:
            # 3D+ case: reshape to [B, T, D] format
            B = x.shape[0]
            Din = x.shape[-1]
            Dout = grad_out.shape[-1]
            
            # Reshape to 3D (views don't allocate new memory)
            x_3d = x.view(B, -1, Din)
            g_3d = grad_out.view(B, -1, Dout)
            
            # Apply per-sample scaling using in-place operation
            coef_view = coef_with_scale.view(-1, 1, 1).to(device=g_3d.device, dtype=g_3d.dtype)
            g_3d.mul_(coef_view)
            
            # grad_w = g_3d^T @ x_3d (flattened)
            # g_3d has been modified in-place with clipping coefficients
            grad_w = torch.matmul(
                g_3d.view(-1, Dout).t(),
                x_3d.view(-1, Din)
            )
            
            # grad_b = sum over all dims except last
            grad_b = g_3d.sum(dim=(0, 1)) if self.bias is not None else None
        
        # Set gradients on parameters (ensure dtype matches parameter dtype)
        grad_w = grad_w.to(dtype=self.weight.dtype)
        
        # Handle DTensor case for FSDP compatibility
        if DTENSOR_AVAILABLE and isinstance(self.weight, DTensor):
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
        # Cast input to weight's dtype for mixed precision compatibility (e.g., FSDP with bfloat16)
        if input.dtype != self.weight.dtype:
            input = input.to(self.weight.dtype)
        
        # If no norm buffer or norm computation disabled, use standard F.linear
        if self._norm_buf is None or not self._compute_norms_container['value']:
            return F.linear(input, self.weight, self.bias)
        
        return FusedFlashLinearFn.apply(
            input, 
            self.weight, 
            self.bias, 
            self._norm_buf,
            self._compute_norms_container,
            self._enable_bookkeeping_container,
            self,  # Pass module reference for caching in BK mode
        )
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


def replace_linear_with_fused(module: nn.Module) -> nn.Module:
    """
    Recursively replace all nn.Linear modules with FusedFlashLinear.
    
    Args:
        module: The module to process
        
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
            fused = FusedFlashLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                device=child.weight.device,
                dtype=child.weight.dtype,
            )
            fused.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                fused.bias.data.copy_(child.bias.data)
            setattr(module, name, fused)
        else:
            replace_linear_with_fused(child)
    
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
