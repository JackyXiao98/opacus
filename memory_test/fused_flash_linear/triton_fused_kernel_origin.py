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
Triton Fused Kernel for Weight Gradient and Per-Sample Norm Computation.

This module provides a Triton kernel that fuses:
1. Weight gradient computation: dW = sum_b(G[b]^T @ X[b])
2. Per-sample gradient norm: Norms[b] = ||G[b]^T @ X[b]||_F^2

Advantages:
- Reads X and G only ONCE from HBM (vs 2x in separate passes)
- Computes per-sample norms "on the fly" in registers
- Eliminates memory overhead of materializing per-sample gradients
"""

import torch

# Try to import Triton - it's optional
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None


# ============================================================================
# Triton Kernel Definition (only defined if Triton is available)
# ============================================================================

if TRITON_AVAILABLE:
    @triton.jit
    def _fused_backward_kernel(
        # Pointers
        X_ptr, G_ptr, DW_ptr, Norms_ptr,
        # Dimensions
        B, T, Din, Dout,
        # Strides
        stride_x_b, stride_x_t, stride_x_d,
        stride_g_b, stride_g_t, stride_g_d,
        stride_dw_out, stride_dw_in,
        stride_norms_b,
        # Meta-parameters
        BLOCK_M: tl.constexpr,  # Tile size for Din
        BLOCK_N: tl.constexpr,  # Tile size for Dout
        BLOCK_K: tl.constexpr,  # Tile size for T (Sequence reduction)
    ):
        """
        Fused kernel that computes weight gradients and per-sample norms.
        
        Grid: (Din / BLOCK_M, Dout / BLOCK_N)
        
        For each tile of the output weight gradient:
        1. Iterate over batch dimension
        2. For each batch, compute the tile of per-sample gradient
        3. Accumulate to global gradient and compute norm contribution
        """
        # Map PID to tile indices
        pid_m = tl.program_id(0)  # Index for Din
        pid_n = tl.program_id(1)  # Index for Dout
        
        # Prepare offsets for output matrix tiles
        # DW is [Dout, Din] (PyTorch Linear weight shape)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # Din range
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # Dout range
        
        # Mask for boundary checks
        mask_m = offs_m < Din
        mask_n = offs_n < Dout
        
        # Initialize accumulator for GLOBAL gradient (sum over B)
        acc_global = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        
        # Outer loop: Iterate over batch dimension
        for b in range(B):
            # Initialize accumulator for PER-SAMPLE gradient (dW_b)
            acc_b = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            
            # Pointers to current batch's X and G
            # X: [B, T, Din], G: [B, T, Dout]
            x_base = X_ptr + b * stride_x_b
            g_base = G_ptr + b * stride_g_b
            
            # Inner loop: Reduce over sequence dimension T
            # Compute dW_b = G_b^T @ X_b
            # Shape: [Dout, T] @ [T, Din] -> [Dout, Din]
            for k in range(0, T, BLOCK_K):
                offs_k = k + tl.arange(0, BLOCK_K)
                mask_k = offs_k < T
                
                # Load X tile: [BLOCK_K, BLOCK_M] (T, Din)
                x_ptrs = x_base + (offs_k[:, None] * stride_x_t + offs_m[None, :] * stride_x_d)
                x_tile = tl.load(x_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
                
                # Load G tile: [BLOCK_K, BLOCK_N] (T, Dout)
                g_ptrs = g_base + (offs_k[:, None] * stride_g_t + offs_n[None, :] * stride_g_d)
                g_tile = tl.load(g_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
                
                # Cast to float32 for numerical stability in dot product
                x_tile = x_tile.to(tl.float32)
                g_tile = g_tile.to(tl.float32)
                
                # Compute partial dW_b: g.T @ x -> [N, K] @ [K, M] -> [N, M]
                # Use tl.trans() for compatibility with older Triton versions
                g_tile_t = tl.trans(g_tile)  # [K, N] -> [N, K]
                acc_b = tl.dot(g_tile_t, x_tile, acc_b)
            
            # --- Per-Batch Processing ---
            
            # A. Accumulate to global gradient (for optimizer)
            acc_global += acc_b
            
            # B. Compute norm contribution (for DP clipping)
            # Mask out padding elements before squaring
            mask_tile = mask_n[:, None] & mask_m[None, :]
            valid_acc_b = tl.where(mask_tile, acc_b, 0.0)
            
            # Sum of squares for this tile
            norm_tile = tl.sum(valid_acc_b * valid_acc_b)
            
            # Atomic add to Norms[b] buffer
            norm_ptr = Norms_ptr + b * stride_norms_b
            tl.atomic_add(norm_ptr, norm_tile)
        
        # Store global gradient
        # DW: [Dout, Din], acc_global: [BLOCK_N, BLOCK_M]
        offs_dw = offs_n[:, None] * stride_dw_out + offs_m[None, :] * stride_dw_in
        tl.store(DW_ptr + offs_dw, acc_global, mask=mask_n[:, None] & mask_m[None, :])


def fused_backward_weight(
    x: torch.Tensor,        # [B, T, Din]
    grad_out: torch.Tensor, # [B, T, Dout]
    norms_buf: torch.Tensor # [B]
) -> torch.Tensor:
    """
    Compute weight gradient AND accumulate per-sample gradient norms in one pass.
    
    This function fuses two operations:
    1. grad_weight = sum_b(grad_out[b]^T @ x[b])  - Standard weight gradient
    2. norms_buf[b] += ||grad_out[b]^T @ x[b]||_F^2  - Per-sample norm squared
    
    Args:
        x: Input activations [B, T, Din] - must be contiguous
        grad_out: Output gradients [B, T, Dout] - must be contiguous
        norms_buf: Buffer to accumulate per-sample squared norms [B]
        
    Returns:
        grad_weight: Weight gradient [Dout, Din]
        
    Note:
        - norms_buf is modified in-place (atomic adds)
        - Requires CUDA and Triton to be available
        - Inputs must be 3D tensors (batch, seq, feature)
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "Triton is not available. Install with: pip install triton"
        )
    
    if not x.is_cuda:
        raise RuntimeError("fused_backward_weight requires CUDA tensors")
    
    B, T, Din = x.shape
    _, _, Dout = grad_out.shape
    
    # Ensure contiguous for Triton
    x_c = x if x.is_contiguous() else x.contiguous()
    g_c = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
    
    # Output gradient weight - always float32 for numerical stability
    grad_weight = torch.empty((Dout, Din), device=x.device, dtype=torch.float32)
    
    # Tuning config - 64x64 tiles are usually good for modern GPUs
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    
    grid = (
        triton.cdiv(Din, BLOCK_M),
        triton.cdiv(Dout, BLOCK_N),
    )
    
    _fused_backward_kernel[grid](
        x_c, g_c, grad_weight, norms_buf,
        B, T, Din, Dout,
        x_c.stride(0), x_c.stride(1), x_c.stride(2),
        g_c.stride(0), g_c.stride(1), g_c.stride(2),
        grad_weight.stride(0), grad_weight.stride(1),
        norms_buf.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    
    return grad_weight


def fused_backward_weight_2d(
    x: torch.Tensor,        # [B, Din]
    grad_out: torch.Tensor, # [B, Dout]
    norms_buf: torch.Tensor # [B]
) -> torch.Tensor:
    """
    Compute weight gradient AND accumulate per-sample gradient norms for 2D inputs.
    
    For 2D case, the per-sample gradient is an outer product (rank-1 matrix):
    grad_weight_i = grad_out_i @ x_i^T
    ||grad_weight_i||_F^2 = ||grad_out_i||^2 * ||x_i||^2
    
    This is more efficient than the 3D case and doesn't require Triton.
    
    Args:
        x: Input activations [B, Din]
        grad_out: Output gradients [B, Dout]
        norms_buf: Buffer to accumulate per-sample squared norms [B]
        
    Returns:
        grad_weight: Weight gradient [Dout, Din]
    """
    # Standard weight gradient: [Dout, B] @ [B, Din] -> [Dout, Din]
    grad_weight = grad_out.t().matmul(x)
    
    # Per-sample norm: ||g_i||^2 * ||x_i||^2 (rank-1 outer product property)
    g_sq = grad_out.pow(2).sum(dim=1)  # [B]
    x_sq = x.pow(2).sum(dim=1)         # [B]
    norms_buf.add_(g_sq * x_sq)
    
    return grad_weight


__all__ = [
    "TRITON_AVAILABLE",
    "fused_backward_weight",
    "fused_backward_weight_2d",
]

