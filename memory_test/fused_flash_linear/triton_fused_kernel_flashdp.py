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
- RuntimeAutoTuner: Dynamically selects between Triton and JIT implementations
- Batch blocking (BK): Memory-efficient processing of large batches
"""

import time
from typing import Optional, Callable, List, Any, Tuple

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
# RuntimeAutoTuner - Ported from flashdp
# ============================================================================

class RuntimeAutoTuner:
    """
    Runtime auto-tuner that dynamically selects the best implementation.
    
    Unlike compile-time autotuning (Triton's @autotune), this class measures
    actual execution time at runtime and caches the best function.
    
    Usage:
        tuner = RuntimeAutoTuner(enable=True)
        best_func = tuner.choose_function([func1, func2], *args, **kwargs)
        result = best_func(*args, **kwargs)
        tuner.final_tune()  # Lock in the choice
    """
    
    def __init__(
        self, 
        enable: bool = True, 
        warmup_iterations: int = 10, 
        measure_iterations: int = 100, 
        log: bool = False
    ) -> None:
        """
        Args:
            enable: If False, always returns the first function (no tuning)
            warmup_iterations: Number of warmup runs before measuring
            measure_iterations: Number of runs for timing measurement
            log: If True, prints the chosen function
        """
        self.enable = enable
        self.if_final_tune = False
        self.chosen_func: Optional[Callable] = None
        self.warmup_iterations = warmup_iterations
        self.measure_iterations = measure_iterations
        self.log = log

    def choose_function(
        self, 
        funcs_list: List[Callable], 
        *args, 
        **kwargs
    ) -> Callable:
        """
        Select the fastest function from the list.
        
        Args:
            funcs_list: List of callable functions with same signature
            *args, **kwargs: Arguments to pass to each function for benchmarking
            
        Returns:
            The fastest function (or cached choice if final_tune() was called)
        """
        if not self.enable:
            return funcs_list[0]
        if self.chosen_func is not None and self.if_final_tune:
            return self.chosen_func
        
        time_list = []
        for func in funcs_list:
            # Warmup
            for _ in range(self.warmup_iterations):
                func(*args, **kwargs)
            # Measure
            time_list.append(self._measure_time(func, *args, **kwargs))
        
        self.chosen_func = funcs_list[time_list.index(min(time_list))]
        if self.log:
            print(f"RuntimeAutoTuner: Chosen function: {self.chosen_func.__name__}")
        return self.chosen_func

    def final_tune(self) -> None:
        """Lock in the current choice, preventing further tuning."""
        self.if_final_tune = True

    def reset(self) -> None:
        """Reset the tuner state, allowing re-tuning."""
        self.if_final_tune = False
        self.chosen_func = None

    def _measure_time(self, func: Callable, *args, **kwargs) -> float:
        """Measure average execution time of a function."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(self.measure_iterations):
            func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        return (end - start) / self.measure_iterations


# Global tuner instance (can be replaced by user)
_global_tuner: Optional[RuntimeAutoTuner] = None


def get_global_tuner() -> RuntimeAutoTuner:
    """Get or create the global RuntimeAutoTuner instance."""
    global _global_tuner
    if _global_tuner is None:
        _global_tuner = RuntimeAutoTuner(enable=True, log=False)
    return _global_tuner


def set_global_tuner(tuner: RuntimeAutoTuner) -> None:
    """Set a custom global RuntimeAutoTuner instance."""
    global _global_tuner
    _global_tuner = tuner


# ============================================================================
# Autotuning Configs - Extended with smaller block sizes
# ============================================================================

def get_autotune_configs():
    """
    Returns a list of configurations to search for the best performance.
    
    Strategies:
    - Large tiles (128x128) for high arithmetic intensity on A100/H100.
    - Smaller tiles (64x64, 32x64) for better occupancy on smaller shapes.
    - Varying num_stages to balance shared memory usage vs. prefetching.
    - Multiple BLOCK_K values for different sequence lengths.
    """
    if not TRITON_AVAILABLE:
        return []
    
    return [
        # === High-performance configs for large matrices (A100/H100) ===
        # Configuration 1: Optimized for H100 with TMA (high num_stages for pipelining)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=8),
        # Configuration 2: Alternative with BLOCK_K=64 for longer sequences
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        # Configuration 3: High-throughput with larger K blocks
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        
        # === Medium configs for mid-range GPUs (3090/4090) ===
        # Configuration 4: Smaller N tile with high stages, good if Dout is small
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # Configuration 5: Balanced for mid-range GPUs
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # Configuration 6: Square medium tiles
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        
        # === Small configs for small matrices / older GPUs ===
        # Configuration 7: Asymmetric small tiles
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        # Configuration 8: Smallest tiles for very small matrices
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 16, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        
        # === Configs with BLOCK_K=16 for short sequences ===
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 16, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ]


# ============================================================================
# JIT Fallback Implementation (torch.jit.script)
# ============================================================================

@torch.jit.script
def _fused_backward_weight_jit_impl(
    x: torch.Tensor,        # [B, T, Din]
    grad_out: torch.Tensor, # [B, T, Dout]
    norms_buf: torch.Tensor, # [B]
    batch_block_size: int = 1
) -> torch.Tensor:
    """
    JIT-compiled fallback for weight gradient + per-sample norm computation.
    
    This implementation uses torch.einsum for the computation and is portable
    across all PyTorch backends (CPU, CUDA without Triton).
    
    Args:
        x: Input activations [B, T, Din]
        grad_out: Output gradients [B, T, Dout]
        norms_buf: Buffer to accumulate per-sample squared norms [B]
        batch_block_size: Number of samples to process together (memory vs speed tradeoff)
        
    Returns:
        grad_weight: Weight gradient [Dout, Din]
    """
    B, T, Din = x.shape
    Dout = grad_out.shape[2]
    
    # Initialize output
    grad_weight = torch.zeros(Dout, Din, device=x.device, dtype=x.dtype)
    
    # Process in blocks for memory efficiency
    for b_start in range(0, B, batch_block_size):
        b_end = min(b_start + batch_block_size, B)
        
        # Slice current block
        x_block = x[b_start:b_end]       # [BK, T, Din]
        g_block = grad_out[b_start:b_end] # [BK, T, Dout]
        
        # Compute per-sample gradients: G[b]^T @ X[b] for each b
        # Using einsum: 'btd,bti->bdi' means [B,T,Dout]^T @ [B,T,Din] -> [B,Dout,Din]
        per_sample_grads = torch.einsum('btd,bti->bdi', g_block, x_block)  # [BK, Dout, Din]
        
        # Compute per-sample norms (squared Frobenius norm)
        per_sample_norms_sq = per_sample_grads.pow(2).sum(dim=(1, 2))  # [BK]
        norms_buf[b_start:b_end] += per_sample_norms_sq
        
        # Accumulate to global gradient
        grad_weight += per_sample_grads.sum(dim=0)
    
    return grad_weight


def fused_backward_weight_jit(
    x: torch.Tensor,        # [B, T, Din]
    grad_out: torch.Tensor, # [B, T, Dout]
    norms_buf: torch.Tensor, # [B]
    batch_block_size: int = 1
) -> torch.Tensor:
    """
    JIT-compiled fallback wrapper (non-JIT entry point).
    
    This is useful when Triton is not available or for debugging.
    """
    return _fused_backward_weight_jit_impl(x, grad_out, norms_buf, batch_block_size)


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
        # Batch block parameters
        B_start, B_end,
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
        1. Iterate over batch dimension (from B_start to B_end)
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
        
        # Outer loop: Iterate over batch dimension (supports batch blocking)
        for b in range(B_start, B_end):
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
    norms_buf: torch.Tensor, # [B]
    batch_block_size: Optional[int] = None
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
        batch_block_size: Optional. Number of batch samples to process per kernel launch.
                         If None, processes all batches in one kernel. Useful for 
                         reducing memory usage with very large batches.
        
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
    grad_weight = torch.zeros((Dout, Din), device=x.device, dtype=torch.float32)
    
    # Default: process all batches together
    if batch_block_size is None:
        batch_block_size = B
    
    # Grid is dynamically calculated based on autotuned block sizes
    grid = lambda META: (
        triton.cdiv(Din, META['BLOCK_M']) * triton.cdiv(Dout, META['BLOCK_N']),
    )
    
    # Process in batch blocks for memory efficiency
    for b_start in range(0, B, batch_block_size):
        b_end = min(b_start + batch_block_size, B)
        
        # Temporary buffer for this block's gradient contribution
        if b_start == 0:
            # First block: write directly to grad_weight
            block_grad_weight = grad_weight
        else:
            # Subsequent blocks: accumulate
            block_grad_weight = torch.zeros((Dout, Din), device=x.device, dtype=torch.float32)
        
        # Kernel launch - autotuner injects optimal BLOCK_M, BLOCK_N, etc.
        _fused_backward_kernel[grid](
            x, grad_out, block_grad_weight, norms_buf,
            B, T, Din, Dout,
            b_start, b_end,
            x.stride(0), x.stride(1), x.stride(2),
            grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
            block_grad_weight.stride(0), block_grad_weight.stride(1),
            norms_buf.stride(0),
        )
        
        # Accumulate gradient from subsequent blocks
        if b_start > 0:
            grad_weight += block_grad_weight
    
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


# ============================================================================
# Bias Gradient with Per-Sample Norm
# ============================================================================

def fused_backward_bias_3d(
    grad_out: torch.Tensor,  # [B, T, Dout]
    norms_buf: torch.Tensor  # [B]
) -> torch.Tensor:
    """
    Compute bias gradient AND accumulate per-sample gradient norms for 3D inputs.
    
    For bias, the per-sample gradient is the sum over the sequence dimension:
    grad_bias_i = grad_out[i].sum(dim=0)  # [Dout]
    ||grad_bias_i||^2 = ||grad_out[i].sum(dim=0)||^2
    
    Args:
        grad_out: Output gradients [B, T, Dout]
        norms_buf: Buffer to accumulate per-sample squared norms [B]
        
    Returns:
        grad_bias: Bias gradient [Dout]
    """
    # Per-sample bias gradient: sum over sequence dimension
    per_sample_bias = grad_out.sum(dim=1)  # [B, Dout]
    
    # Global bias gradient: sum over batch
    grad_bias = per_sample_bias.sum(dim=0)  # [Dout]
    
    # Per-sample norm: ||grad_bias_i||^2
    per_sample_norms_sq = per_sample_bias.pow(2).sum(dim=1)  # [B]
    norms_buf.add_(per_sample_norms_sq)
    
    return grad_bias


def fused_backward_bias_2d(
    grad_out: torch.Tensor,  # [B, Dout]
    norms_buf: torch.Tensor  # [B]
) -> torch.Tensor:
    """
    Compute bias gradient AND accumulate per-sample gradient norms for 2D inputs.
    
    For 2D case, per-sample gradient IS grad_out[i]:
    grad_bias_i = grad_out[i]  # [Dout]
    ||grad_bias_i||^2 = ||grad_out[i]||^2
    
    Args:
        grad_out: Output gradients [B, Dout]
        norms_buf: Buffer to accumulate per-sample squared norms [B]
        
    Returns:
        grad_bias: Bias gradient [Dout]
    """
    # Global bias gradient: sum over batch
    grad_bias = grad_out.sum(dim=0)  # [Dout]
    
    # Per-sample norm: ||grad_out[i]||^2
    per_sample_norms_sq = grad_out.pow(2).sum(dim=1)  # [B]
    norms_buf.add_(per_sample_norms_sq)
    
    return grad_bias


# ============================================================================
# Unified Entry Point
# ============================================================================

def fused_backward_linear(
    x: torch.Tensor,              # [B, T, Din] or [B, Din]
    grad_out: torch.Tensor,       # [B, T, Dout] or [B, Dout]
    norms_buf_weight: torch.Tensor,  # [B]
    norms_buf_bias: Optional[torch.Tensor] = None,  # [B] or None
    has_bias: bool = False,
    batch_block_size: Optional[int] = None,
    use_triton: Optional[bool] = None,
    tuner: Optional[RuntimeAutoTuner] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Unified entry point for computing linear layer gradients with per-sample norms.
    
    This function:
    1. Automatically handles 2D and 3D inputs
    2. Optionally uses RuntimeAutoTuner to select Triton vs JIT implementation
    3. Computes both weight and bias gradients if needed
    4. Supports batch blocking for memory efficiency
    
    Args:
        x: Input activations [B, T, Din] (3D) or [B, Din] (2D)
        grad_out: Output gradients [B, T, Dout] (3D) or [B, Dout] (2D)
        norms_buf_weight: Buffer to accumulate per-sample weight gradient norms [B]
        norms_buf_bias: Buffer to accumulate per-sample bias gradient norms [B] (optional)
        has_bias: Whether to compute bias gradient
        batch_block_size: Number of batch samples per kernel launch (memory vs speed)
        use_triton: Force Triton (True), force JIT (False), or auto-select (None)
        tuner: Optional RuntimeAutoTuner for dynamic implementation selection
        
    Returns:
        Tuple of (grad_weight, grad_bias) where grad_bias is None if has_bias=False
        
    Example:
        >>> x = torch.randn(32, 128, 768, device='cuda')  # [B, T, Din]
        >>> grad_out = torch.randn(32, 128, 1024, device='cuda')  # [B, T, Dout]
        >>> norms_w = torch.zeros(32, device='cuda')
        >>> norms_b = torch.zeros(32, device='cuda')
        >>> grad_w, grad_b = fused_backward_linear(x, grad_out, norms_w, norms_b, has_bias=True)
    """
    is_3d = x.dim() == 3
    
    # Determine whether to use Triton
    if use_triton is None:
        # Auto-select: use Triton if available and on CUDA
        use_triton = TRITON_AVAILABLE and x.is_cuda and is_3d
    
    # Compute weight gradient
    if is_3d:
        if use_triton and TRITON_AVAILABLE:
            # Use Triton kernel
            if tuner is not None:
                # Use RuntimeAutoTuner to select between Triton and JIT
                def _triton_impl():
                    return fused_backward_weight(x, grad_out, norms_buf_weight, batch_block_size)
                
                def _jit_impl():
                    bk = batch_block_size if batch_block_size is not None else 1
                    return fused_backward_weight_jit(x, grad_out, norms_buf_weight, bk)
                
                # Choose best implementation
                best_func = tuner.choose_function([_triton_impl, _jit_impl])
                grad_weight = best_func()
            else:
                grad_weight = fused_backward_weight(x, grad_out, norms_buf_weight, batch_block_size)
        else:
            # Use JIT fallback
            bk = batch_block_size if batch_block_size is not None else 1
            grad_weight = fused_backward_weight_jit(x, grad_out, norms_buf_weight, bk)
    else:
        # 2D case - always use optimized PyTorch implementation
        grad_weight = fused_backward_weight_2d(x, grad_out, norms_buf_weight)
    
    # Compute bias gradient if needed
    grad_bias = None
    if has_bias:
        if norms_buf_bias is None:
            raise ValueError("norms_buf_bias must be provided when has_bias=True")
        
        if is_3d:
            grad_bias = fused_backward_bias_3d(grad_out, norms_buf_bias)
        else:
            grad_bias = fused_backward_bias_2d(grad_out, norms_buf_bias)
    
    return grad_weight, grad_bias


__all__ = [
    # Availability flag
    "TRITON_AVAILABLE",
    # RuntimeAutoTuner
    "RuntimeAutoTuner",
    "get_global_tuner",
    "set_global_tuner",
    # Core functions
    "fused_backward_weight",
    "fused_backward_weight_2d",
    "fused_backward_weight_jit",
    # Bias functions
    "fused_backward_bias_3d",
    "fused_backward_bias_2d",
    # Unified entry point
    "fused_backward_linear",
]
