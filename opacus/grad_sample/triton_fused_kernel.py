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
# Hardware Detection Utilities
# ============================================================================

# Cache for hardware detection results
_HOPPER_GPU_CACHE = None
_DSMEM_SUPPORT_CACHE = None


def is_hopper_gpu() -> bool:
    """
    Check if current GPU supports DSMEM (H100/Hopper, compute capability >= 9.0).
    
    DSMEM (Distributed Shared Memory) is a Hopper-specific feature that allows
    direct SM-to-SM communication via Thread Block Clusters.
    
    Returns:
        True if running on H100/Hopper GPU with compute capability >= 9.0
    """
    global _HOPPER_GPU_CACHE
    
    if _HOPPER_GPU_CACHE is not None:
        return _HOPPER_GPU_CACHE
    
    if not torch.cuda.is_available():
        _HOPPER_GPU_CACHE = False
        return False
    
    try:
        # Get compute capability of current device
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        # Hopper (H100) has compute capability 9.0
        _HOPPER_GPU_CACHE = major >= 9
    except Exception:
        _HOPPER_GPU_CACHE = False
    
    return _HOPPER_GPU_CACHE


def has_dsmem_support() -> bool:
    """
    Check if Triton version supports cluster/DSMEM APIs.
    
    DSMEM requires:
    1. Triton >= 2.1.0 (experimental cluster support)
    2. tl.extra.cuda module with mbarrier primitives
    
    Returns:
        True if Triton has DSMEM/cluster support
    """
    global _DSMEM_SUPPORT_CACHE
    
    if _DSMEM_SUPPORT_CACHE is not None:
        return _DSMEM_SUPPORT_CACHE
    
    if not TRITON_AVAILABLE:
        _DSMEM_SUPPORT_CACHE = False
        return False
    
    try:
        # Check Triton version (need >= 2.1.0 for cluster support)
        triton_version = getattr(triton, '__version__', '0.0.0')
        version_parts = triton_version.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
        
        if major < 2 or (major == 2 and minor < 1):
            _DSMEM_SUPPORT_CACHE = False
            return False
        
        # Check for cluster_dims support in triton.Config
        # and mbarrier primitives in tl.extra.cuda
        has_cluster = hasattr(triton, 'Config')
        has_extra = hasattr(tl, 'extra') and hasattr(tl.extra, 'cuda')
        
        _DSMEM_SUPPORT_CACHE = has_cluster and has_extra
    except Exception:
        _DSMEM_SUPPORT_CACHE = False
    
    return _DSMEM_SUPPORT_CACHE


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


def get_dsmem_autotune_configs():
    """
    Returns autotuning configurations for DSMEM-enabled kernel on H100.
    
    Key differences from standard configs:
    - SPLIT_K: Number of blocks in cluster that split the T dimension
    - cluster_dims: Thread Block Cluster dimensions (SPLIT_K, 1, 1)
    - Smaller tile sizes to fit multiple partial results in SRAM
    
    SRAM Budget Considerations:
    - H100 has 228KB shared memory per SM
    - Each partial result tile needs BLOCK_M * BLOCK_N * 4 bytes (FP32)
    - For SPLIT_K blocks, leader needs (SPLIT_K-1) buffers in SRAM
    - 64x64 tile = 16KB, 3 buffers = 48KB (safe for SPLIT_K=4)
    - 64x128 tile = 32KB, 1 buffer = 32KB (safe for SPLIT_K=2)
    """
    if not TRITON_AVAILABLE:
        return []
    
    configs = []
    
    # Configuration 1: Split-2, larger tiles (safer SRAM, higher compute per block)
    # SRAM: leader stores 1 partial = 64*128*4 = 32KB
    configs.append(triton.Config(
        {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 2},
        num_stages=4, num_warps=4,
        pre_hook=None,  # cluster_dims set at launch time
    ))
    
    # Configuration 2: Split-2, balanced tiles
    # SRAM: leader stores 1 partial = 64*64*4 = 16KB
    configs.append(triton.Config(
        {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 2},
        num_stages=4, num_warps=4,
    ))
    
    # Configuration 3: Split-4, smaller tiles (higher parallelism)
    # SRAM: leader stores 3 partials = 64*64*4*3 = 48KB (safe)
    configs.append(triton.Config(
        {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4},
        num_stages=3, num_warps=4,
    ))
    
    # Configuration 4: Split-4, even smaller tiles for very large T
    # SRAM: leader stores 3 partials = 32*64*4*3 = 24KB (very safe)
    configs.append(triton.Config(
        {'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4},
        num_stages=3, num_warps=4,
    ))
    
    # Configuration 5: Split-2 with larger tiles for compute-bound cases
    # SRAM: leader stores 1 partial = 128*64*4 = 32KB
    configs.append(triton.Config(
        {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 2},
        num_stages=4, num_warps=8,
    ))
    
    return configs


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


# ============================================================================
# Split-K Kernel for Parallel T-Dimension Reduction
#
# This kernel splits the sequence dimension T across multiple thread blocks,
# with proper atomic barrier synchronization for inter-block communication.
#
# Key Features:
# - Split-T: Divides sequence dimension T across SPLIT_K blocks
# - Atomic Barriers: Uses atomic counters for proper inter-block synchronization
# - Global Memory Buffer: Stores partial results for cross-block reduction
# - Leader Aggregation: Block 0 in each tile group aggregates all partials
#
# Note: This is a portable implementation that works on all CUDA GPUs.
# For true H100 DSMEM with on-chip reduction, hardware cluster support
# with mbarrier primitives would be needed.
# ============================================================================

if TRITON_AVAILABLE:
    @triton.jit
    def _fused_backward_kernel_dsmem(
        # Pointers
        X_ptr, G_ptr, DW_ptr, Norms_ptr,
        # Buffer for inter-block communication
        Partial_buf_ptr,
        # Barrier buffer for synchronization [num_tiles]
        Barrier_ptr,
        # Dimensions
        B, T, Din, Dout,
        # Strides
        stride_x_b, stride_x_t, stride_x_d,
        stride_g_b, stride_g_t, stride_g_d,
        stride_dw_out, stride_dw_in,
        stride_norms_b,
        # Partial buffer strides
        stride_partial_tile, stride_partial_slot, stride_partial_n, stride_partial_m,
        # Meta-parameters
        BLOCK_M: tl.constexpr,  # Tile size for Din
        BLOCK_N: tl.constexpr,  # Tile size for Dout
        BLOCK_K: tl.constexpr,  # Tile size for T (Sequence reduction)
        GROUP_SIZE_M: tl.constexpr,  # Group size for L2 Swizzling
        SPLIT_K: tl.constexpr,  # Number of blocks per tile (T split factor)
    ):
        """
        Split-K fused kernel with atomic barrier synchronization.
        
        Architecture:
        - Grid is launched with SPLIT_K blocks per output tile
        - Each block computes partial gradient for its T slice
        - Blocks synchronize via atomic counters in global memory
        - Leader block (cluster_id=0) aggregates and writes final result
        
        Synchronization Protocol:
        1. All blocks compute their local partial gradient (T slice)
        2. Non-leaders write partial to global buffer, then atomically signal
        3. Leader spins on atomic counter until all non-leaders have signaled
        4. Leader reads partials, aggregates, computes norm, resets barrier
        5. All blocks wait for leader to finish before next batch
        """
        # --- Block Identity ---
        pid = tl.program_id(axis=0)
        
        # Tile layout
        num_pid_m = tl.cdiv(Din, BLOCK_M)
        num_pid_n = tl.cdiv(Dout, BLOCK_N)
        num_tiles = num_pid_m * num_pid_n
        
        tile_id = pid // SPLIT_K  # Which output tile
        cluster_id = pid % SPLIT_K  # Position within tile group (0 = leader)
        
        # Early exit if tile_id is out of bounds
        if tile_id >= num_tiles:
            return
        
        # --- L2 Cache Swizzling for tile_id ---
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        
        local_tile_id = tile_id % num_pid_in_group
        pid_m = first_pid_m + (local_tile_id % group_size_m)
        pid_n = local_tile_id // group_size_m
        
        # Output tile offsets
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # Din range
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # Dout range
        
        # Boundary masks
        mask_m = offs_m < Din
        mask_n = offs_n < Dout
        mask_tile = mask_n[:, None] & mask_m[None, :]
        
        # --- Split-T Workload Distribution ---
        t_per_block = tl.cdiv(T, SPLIT_K)
        t_start = cluster_id * t_per_block
        t_end = min(t_start + t_per_block, T)
        
        # Barrier pointer for this tile
        barrier_ptr = Barrier_ptr + tile_id
        
        # Initialize accumulator for GLOBAL gradient (sum over B)
        acc_global = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        
        # Outer loop: Iterate over batch dimension
        for b in range(B):
            # Initialize accumulator for PER-SAMPLE partial gradient
            acc_partial = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            
            # --- TMA Block Pointers for current batch (T slice) ---
            x_block_ptr = tl.make_block_ptr(
                base=X_ptr + b * stride_x_b,
                shape=(T, Din),
                strides=(stride_x_t, stride_x_d),
                offsets=(t_start, pid_m * BLOCK_M),
                block_shape=(BLOCK_K, BLOCK_M),
                order=(1, 0)
            )
            
            g_block_ptr = tl.make_block_ptr(
                base=G_ptr + b * stride_g_b,
                shape=(T, Dout),
                strides=(stride_g_t, stride_g_d),
                offsets=(t_start, pid_n * BLOCK_N),
                block_shape=(BLOCK_K, BLOCK_N),
                order=(1, 0)
            )
            
            # Inner loop: Reduce over this block's T slice
            for k in range(t_start, t_end, BLOCK_K):
                x_tile = tl.load(x_block_ptr, boundary_check=(0, 1))
                g_tile = tl.load(g_block_ptr, boundary_check=(0, 1))
                acc_partial = tl.dot(tl.trans(g_tile), x_tile, acc_partial, allow_tf32=True)
                x_block_ptr = tl.advance(x_block_ptr, (BLOCK_K, 0))
                g_block_ptr = tl.advance(g_block_ptr, (BLOCK_K, 0))
            
            # === Inter-Block Reduction with Atomic Barrier ===
            
            if cluster_id != 0:
                # --- Non-Leader: Write partial and signal ---
                slot = cluster_id - 1
                partial_offs = (
                    Partial_buf_ptr +
                    tile_id * stride_partial_tile +
                    slot * stride_partial_slot +
                    tl.arange(0, BLOCK_N)[:, None] * stride_partial_n +
                    tl.arange(0, BLOCK_M)[None, :] * stride_partial_m
                )
                
                # Store partial to global buffer
                tl.store(partial_offs, acc_partial, mask=mask_tile)
                
                # Memory fence (atomic_add acts as release fence)
                # Signal leader that this block's partial is ready
                tl.atomic_add(barrier_ptr, 1)
                
                # Wait for leader to finish aggregation (barrier reset to 0)
                # This prevents race condition on next batch iteration
                while tl.atomic_add(barrier_ptr, 0) > 0:
                    pass
            
            else:
                # --- Leader: Wait, aggregate, compute norm, reset barrier ---
                
                # Wait for all non-leaders to signal (counter == SPLIT_K - 1)
                expected_count = SPLIT_K - 1
                while tl.atomic_add(barrier_ptr, 0) < expected_count:
                    pass
                
                # Aggregate partials from all non-leaders
                acc_b = acc_partial  # Start with leader's own partial
                
                for s in range(SPLIT_K - 1):
                    partial_offs = (
                        Partial_buf_ptr +
                        tile_id * stride_partial_tile +
                        s * stride_partial_slot +
                        tl.arange(0, BLOCK_N)[:, None] * stride_partial_n +
                        tl.arange(0, BLOCK_M)[None, :] * stride_partial_m
                    )
                    partial = tl.load(partial_offs, mask=mask_tile, other=0.0)
                    acc_b += partial
                
                # Accumulate to global gradient
                acc_global += acc_b
                
                # Compute norm contribution
                acc_b_sq = acc_b * acc_b
                valid_acc_b_sq = tl.where(mask_tile, acc_b_sq, 0.0)
                norm_tile = tl.sum(valid_acc_b_sq)
                
                # Atomic add to Norms[b] buffer
                norm_ptr = Norms_ptr + b * stride_norms_b
                tl.atomic_add(norm_ptr, norm_tile)
                
                # Reset barrier for next batch iteration
                # (atomic_xchg ensures memory fence)
                tl.atomic_xchg(barrier_ptr, 0)
        
        # --- Final Output (Leader only) ---
        if cluster_id == 0:
            offs_dw = offs_n[:, None] * stride_dw_out + offs_m[None, :] * stride_dw_in
            tl.store(DW_ptr + offs_dw, acc_global, mask=mask_tile)


def fused_backward_weight(
    x: torch.Tensor,        # [B, T, Din]
    grad_out: torch.Tensor, # [B, T, Dout]
    norms_buf: torch.Tensor, # [B]
    use_dsmem: bool = None,  # Auto-detect if None
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
        use_dsmem: Whether to use DSMEM-optimized kernel for H100.
                   If None (default), auto-detects based on GPU capability.
                   If True, forces DSMEM kernel (falls back if not available).
                   If False, uses standard kernel.
        
    Returns:
        grad_weight: Weight gradient [Dout, Din]
        
    Note:
        - norms_buf is modified in-place (atomic adds)
        - Requires CUDA and Triton to be available
        - Inputs must be 3D tensors (batch, seq, feature)
        - Uses autotuning to select optimal block sizes per input shape
        - On H100 GPUs, automatically uses DSMEM-optimized Split-T kernel
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "Triton is not available. Install with: pip install triton"
        )
    
    if not x.is_cuda:
        raise RuntimeError("fused_backward_weight requires CUDA tensors")
    
    # --- Auto-select Split-K kernel for large T ---
    # Split-K provides speedup for large T by parallelizing the T reduction
    # across multiple thread blocks with atomic barrier synchronization.
    if use_dsmem is None:
        # Auto-enable for large T where parallelism benefits outweigh sync overhead
        T = x.shape[1]
        use_dsmem = T >= 512  # Only use Split-K for large sequences
    
    if use_dsmem:
        # Choose split_k based on sequence length T
        # Larger T benefits from more parallelism (split_k=4)
        # Medium T should use conservative split (split_k=2)
        T = x.shape[1]
        split_k = 4 if T >= 1024 else 2
        return fused_backward_weight_dsmem(x, grad_out, norms_buf, split_k=split_k)
    
    # --- Standard kernel for non-H100 GPUs ---
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


def fused_backward_weight_dsmem(
    x: torch.Tensor,        # [B, T, Din]
    grad_out: torch.Tensor, # [B, T, Dout]
    norms_buf: torch.Tensor, # [B]
    split_k: int = 2,       # Number of T splits (cluster size)
) -> torch.Tensor:
    """
    Split-K optimized weight gradient computation.
    
    Splits the sequence dimension T across multiple thread blocks for
    increased parallelism, with atomic barrier synchronization for
    inter-block communication.
    
    This can provide speedup for large T by enabling better GPU utilization
    through parallel reduction.
    
    Args:
        x: Input activations [B, T, Din] - must be contiguous
        grad_out: Output gradients [B, T, Dout] - must be contiguous
        norms_buf: Buffer to accumulate per-sample squared norms [B]
        split_k: Number of blocks per tile (T split factor, 2 or 4)
        
    Returns:
        grad_weight: Weight gradient [Dout, Din]
        
    Requirements:
        - CUDA tensors
        - Triton available
        
    Note:
        - split_k should be chosen based on T size
        - Larger split_k = more parallelism but more synchronization overhead
        - Best for large T (>= 512)
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "Triton is not available. Install with: pip install triton"
        )
    
    if not x.is_cuda:
        raise RuntimeError("fused_backward_weight_dsmem requires CUDA tensors")
    
    # Validate split_k
    if split_k not in [2, 4]:
        raise ValueError(f"split_k must be 2 or 4, got {split_k}")
    
    B, T, Din = x.shape
    _, _, Dout = grad_out.shape
    
    # For small T, split-K overhead isn't worth it
    if T < 128:
        return fused_backward_weight(x, grad_out, norms_buf, use_dsmem=False)
    
    # Ensure contiguous for vectorized loads
    if not x.is_contiguous():
        x = x.contiguous()
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()
    
    # Output gradient weight - always float32 for numerical stability
    grad_weight = torch.empty((Dout, Din), device=x.device, dtype=torch.float32)
    
    # Tile sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    GROUP_SIZE_M = 8
    
    num_tiles_m = triton.cdiv(Din, BLOCK_M)
    num_tiles_n = triton.cdiv(Dout, BLOCK_N)
    num_tiles = num_tiles_m * num_tiles_n
    
    # Allocate partial buffer for inter-block communication
    # Layout: [num_tiles, split_k-1, BLOCK_N, BLOCK_M]
    partial_buf = torch.empty(
        (num_tiles, split_k - 1, BLOCK_N, BLOCK_M),
        device=x.device,
        dtype=torch.float32
    )
    
    # Allocate barrier buffer for synchronization
    # One int32 counter per tile
    barrier_buf = torch.zeros(
        (num_tiles,),
        device=x.device,
        dtype=torch.int32
    )
    
    # Grid: num_tiles * split_k blocks
    grid = (num_tiles * split_k,)
    
    # Kernel launch
    _fused_backward_kernel_dsmem[grid](
        x, grad_out, grad_weight, norms_buf,
        partial_buf,
        barrier_buf,
        B, T, Din, Dout,
        x.stride(0), x.stride(1), x.stride(2),
        grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
        grad_weight.stride(0), grad_weight.stride(1),
        norms_buf.stride(0),
        partial_buf.stride(0),  # stride_partial_tile
        partial_buf.stride(1),  # stride_partial_slot
        partial_buf.stride(2),  # stride_partial_n
        partial_buf.stride(3),  # stride_partial_m
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        SPLIT_K=split_k,
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
    "is_hopper_gpu",
    "has_dsmem_support",
    "fused_backward_weight",
    "fused_backward_weight_dsmem",
    "fused_backward_weight_2d",
]
