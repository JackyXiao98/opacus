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

Optimizations:
- TMA Block Pointer API: Uses tl.make_block_ptr for H100 TMA acceleration
- Software Pipelining: High num_stages (4-5) for async memory prefetching
- Autotuning: Dynamically selects best block sizes and num_warps
- TF32 Support: Explicitly allows TF32 for Tensor Cores on Ampere+
- L2 Cache Swizzling for better cache hit rates
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
# Autotuning Configs
# ============================================================================

def get_autotune_configs():
    """
    Returns a list of configurations to search for the best performance.
    
    Strategies:
    - Large tiles (128x128) for high arithmetic intensity on A100/H100.
    - Smaller tiles (64x64, 32x64) for better occupancy on smaller shapes.
    - Varying num_stages to balance shared memory usage vs. prefetching.
    """
    if not TRITON_AVAILABLE:
        return []
    
    return [
        # Configuration 1: Optimized for H100 with TMA (high num_stages for pipelining)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=8),
        # Configuration 2: Alternative high-performance config for large matrices
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        # Configuration 3: Smaller N tile with high stages, good if Dout is small
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=4),
        # Configuration 4: Balanced for mid-range GPUs (3090/4090)
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # Configuration 5: Conservative (fallback for small shapes/older GPUs)
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
    ]


# ============================================================================
# Triton Kernel Definition (only defined if Triton is available)
# 
# Optimizations:
# - TMA Block Pointer API (tl.make_block_ptr) for H100 TMA acceleration
# - Software Pipelining via high num_stages for async memory prefetching
# - L2 Cache Swizzling (Grouped Program ID) for better cache hit rates
# - Autotuning for optimal block sizes per input shape
# - TF32 enabled for Tensor Core acceleration on Ampere+
# - Pre-square optimization for better pipelining
# ============================================================================

if TRITON_AVAILABLE:
    @triton.autotune(
        configs=get_autotune_configs(),
        key=['Din', 'Dout', 'T']
    )
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
        # Meta-parameters (Injected by Autotuner)
        BLOCK_M: tl.constexpr,  # Tile size for Din
        BLOCK_N: tl.constexpr,  # Tile size for Dout
        BLOCK_K: tl.constexpr,  # Tile size for T (Sequence reduction)
        GROUP_SIZE_M: tl.constexpr,  # Group size for L2 Swizzling
    ):
        """
        Fused kernel that computes weight gradients and per-sample norms.
        
        Grid: 1D launch with (Din / BLOCK_M) * (Dout / BLOCK_N) programs
        Uses L2 cache swizzling for better memory locality.
        
        For each tile of the output weight gradient:
        1. Iterate over batch dimension
        2. For each batch, compute the tile of per-sample gradient
        3. Accumulate to global gradient and compute norm contribution
        """
        # --- L2 Cache Swizzling Logic ---
        # Map 1D PID to 2D Tiled PID with locality awareness
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(Din, BLOCK_M)
        num_pid_n = tl.cdiv(Dout, BLOCK_N)
        
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        
        # Re-mapped PIDs for better L2 cache locality
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        # --------------------------------
        
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
            
            # --- TMA Block Pointers for current batch ---
            # X: [B, T, Din] -> slice [T, Din] for batch b
            # Block shape: [BLOCK_K, BLOCK_M] (T, Din)
            x_block_ptr = tl.make_block_ptr(
                base=X_ptr + b * stride_x_b,
                shape=(T, Din),
                strides=(stride_x_t, stride_x_d),
                offsets=(0, pid_m * BLOCK_M),
                block_shape=(BLOCK_K, BLOCK_M),
                order=(1, 0)  # Din is contiguous (stride=1)
            )
            
            # G: [B, T, Dout] -> slice [T, Dout] for batch b
            # Block shape: [BLOCK_K, BLOCK_N] (T, Dout)
            g_block_ptr = tl.make_block_ptr(
                base=G_ptr + b * stride_g_b,
                shape=(T, Dout),
                strides=(stride_g_t, stride_g_d),
                offsets=(0, pid_n * BLOCK_N),
                block_shape=(BLOCK_K, BLOCK_N),
                order=(1, 0)  # Dout is contiguous (stride=1)
            )
            
            # Inner loop: Reduce over sequence dimension T
            # Compute dW_b = G_b^T @ X_b
            # Shape: [Dout, T] @ [T, Din] -> [Dout, Din]
            for k in range(0, T, BLOCK_K):
                # Load with TMA - boundary_check handles masking automatically
                # (0, 1) checks both T dimension (0) and feature dimension (1)
                x_tile = tl.load(x_block_ptr, boundary_check=(0, 1))
                g_tile = tl.load(g_block_ptr, boundary_check=(0, 1))
                
                # Compute partial dW_b: g.T @ x -> [N, K] @ [K, M] -> [N, M]
                # TF32 enabled for Tensor Core acceleration on Ampere+ GPUs
                acc_b = tl.dot(tl.trans(g_tile), x_tile, acc_b, allow_tf32=True)
                
                # Advance block pointers along T dimension
                x_block_ptr = tl.advance(x_block_ptr, (BLOCK_K, 0))
                g_block_ptr = tl.advance(g_block_ptr, (BLOCK_K, 0))
            
            # --- Per-Batch Processing ---
            
            # A. Accumulate to global gradient (for optimizer)
            acc_global += acc_b
            
            # B. Compute norm contribution (for DP clipping)
            # Optimization: Pre-square in registers before applying mask
            # This keeps the pipeline busy
            acc_b_sq = acc_b * acc_b
            
            # Apply mask only once at the end
            mask_tile = mask_n[:, None] & mask_m[None, :]
            valid_acc_b_sq = tl.where(mask_tile, acc_b_sq, 0.0)
            
            # Sum of squares for this tile
            norm_tile = tl.sum(valid_acc_b_sq)
            
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
        - Uses autotuning to select optimal block sizes per input shape
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "Triton is not available. Install with: pip install triton"
        )
    
    if not x.is_cuda:
        raise RuntimeError("fused_backward_weight requires CUDA tensors")
    
    B, T, Din = x.shape
    _, _, Dout = grad_out.shape
    
    # Ensure contiguous for vectorized loads
    if not x.is_contiguous():
        x = x.contiguous()
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()
    
    # Output gradient weight - always float32 for numerical stability
    grad_weight = torch.empty((Dout, Din), device=x.device, dtype=torch.float32)
    
    # Grid is dynamically calculated based on autotuned block sizes
    grid = lambda META: (
        triton.cdiv(Din, META['BLOCK_M']) * triton.cdiv(Dout, META['BLOCK_N']),
    )
    
    # Kernel launch - autotuner injects optimal BLOCK_M, BLOCK_N, etc.
    _fused_backward_kernel[grid](
        x, grad_out, grad_weight, norms_buf,
        B, T, Din, Dout,
        x.stride(0), x.stride(1), x.stride(2),
        grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
        grad_weight.stride(0), grad_weight.stride(1),
        norms_buf.stride(0),
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
