"""
Flash Clipping Algorithms for Linear Layers

This module implements two flash clipping algorithms:
1. Input-Length-Linear Algorithm: O(T * d^2) - optimal for long sequences
2. Width-Linear Algorithm: O(T^2 * d) - optimal for wide models

Both algorithms compute per-sample gradient norms efficiently without materializing
the full per-sample gradients.

With optional Triton acceleration for GPU computation.
"""

from typing import Dict, List
import torch
import torch.nn as nn
import time
import os

# Try to import Triton for GPU acceleration
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None


def is_triton_available() -> bool:
    """Check if Triton is available for GPU acceleration."""
    return TRITON_AVAILABLE and torch.cuda.is_available()


def _get_optimal_block_sizes(d_a: int, d_g: int, tile_size: int, algorithm: str):
    """
    Compute optimal block sizes for Triton kernels based on problem dimensions.
    
    This heuristic aims to:
    1. Maximize occupancy (use power-of-2 sizes)
    2. Minimize register spills (keep blocks reasonably small)
    3. Maximize data reuse (balance block dimensions)
    
    Args:
        d_a: Activation dimension
        d_g: Gradient dimension
        tile_size: Time tile size
        algorithm: "input_length" or "width"
    
    Returns:
        Dictionary of optimal block sizes for the given algorithm
    """
    if not TRITON_AVAILABLE:
        return {}
    
    if algorithm == "input_length":
        # For input_length: balance d_a and d_g dimensions
        # Prefer smaller blocks to fit M_j and M_k blocks in registers/shared memory
        BLOCK_D_A = min(64, triton.next_power_of_2(d_a))
        BLOCK_D_G = min(64, triton.next_power_of_2(d_g))
        # Time dimension can be larger since we're computing outer products
        BLOCK_T = min(128, triton.next_power_of_2(tile_size))
        
        return {
            'BLOCK_D_A': BLOCK_D_A,
            'BLOCK_D_G': BLOCK_D_G,
            'BLOCK_T': BLOCK_T,
        }
    
    elif algorithm == "width":
        # For width: balance time and dimension blocks
        # Prefer smaller time blocks to fit Score_a and Score_g blocks in shared memory
        BLOCK_TAU_J = min(64, triton.next_power_of_2(tile_size))
        BLOCK_TAU_K = min(64, triton.next_power_of_2(tile_size))
        # Dimension can be larger since we're reducing over it
        BLOCK_D = min(128, triton.next_power_of_2(max(d_a, d_g)))
        
        return {
            'BLOCK_TAU_J': BLOCK_TAU_J,
            'BLOCK_TAU_K': BLOCK_TAU_K,
            'BLOCK_D': BLOCK_D,
        }
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


# ============================================================================
# Triton Kernels
# ============================================================================

if TRITON_AVAILABLE:
    @triton.jit
    def sum_over_time_norm_squared_kernel(
        # Input tensor
        G_ptr,
        # Output tensor
        output_ptr,
        # Tensor dimensions
        B, T, d_g,
        # Strides
        stride_G_b, stride_G_t, stride_G_d,
        # Block sizes
        BLOCK_SIZE_T: tl.constexpr,
        BLOCK_SIZE_D: tl.constexpr,
    ):
        """
        Triton kernel for computing ||sum_t G[b,t,:]||_2^2.
        This is equivalent to sum_k (sum_t G[b,t,k])^2
        """
        batch_id = tl.program_id(0)
        d_id = tl.program_id(1)
        
        # Calculate dimension range for this block
        d_start = d_id * BLOCK_SIZE_D
        d_end = min(d_start + BLOCK_SIZE_D, d_g)
        d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
        d_mask = d_offsets < d_end
        
        # Initialize sum accumulator for each dimension
        sum_over_time = tl.zeros([BLOCK_SIZE_D], dtype=tl.float32)
        
        # Sum over time dimension
        for t_start in range(0, T, BLOCK_SIZE_T):
            t_end = min(t_start + BLOCK_SIZE_T, T)
            t_offsets = t_start + tl.arange(0, BLOCK_SIZE_T)
            t_mask = t_offsets < t_end
            
            # Load G block
            G_ptrs = G_ptr + batch_id * stride_G_b + t_offsets[:, None] * stride_G_t + d_offsets[None, :] * stride_G_d
            G_block = tl.load(G_ptrs, mask=t_mask[:, None] & d_mask[None, :], other=0.0)
            
            # Sum over time for this dimension block
            time_sum = tl.sum(G_block, axis=0)
            sum_over_time += time_sum
        
        # Compute L2 norm squared for this dimension block
        norm_squared_block = sum_over_time * sum_over_time
        
        # Store partial results
        output_ptrs = output_ptr + batch_id * d_g + d_offsets
        tl.store(output_ptrs, norm_squared_block, mask=d_mask)


    @triton.jit
    def frobenius_inner_product_kernel(
        # Input matrices
        M1_ptr, M2_ptr,
        # Output
        output_ptr,
        # Dimensions
        B, d1, d2,
        # Strides
        stride_M1_b, stride_M1_d1, stride_M1_d2,
        stride_M2_b, stride_M2_d1, stride_M2_d2,
        # Block sizes
        BLOCK_D1: tl.constexpr,
        BLOCK_D2: tl.constexpr,
    ):
        """
        Fused kernel for computing Frobenius inner product: <M1, M2>_F = sum(M1 * M2)
        This avoids the element-wise multiply + sum in PyTorch.
        """
        batch_id = tl.program_id(0)
        
        accumulator = 0.0
        
        # Iterate over d1 dimension in blocks
        for d1_start in range(0, d1, BLOCK_D1):
            d1_end = min(d1_start + BLOCK_D1, d1)
            d1_offsets = d1_start + tl.arange(0, BLOCK_D1)
            d1_mask = d1_offsets < d1_end
            
            # Iterate over d2 dimension in blocks
            for d2_start in range(0, d2, BLOCK_D2):
                d2_end = min(d2_start + BLOCK_D2, d2)
                d2_offsets = d2_start + tl.arange(0, BLOCK_D2)
                d2_mask = d2_offsets < d2_end
                
                # Load M1 block
                M1_ptrs = (M1_ptr + batch_id * stride_M1_b + 
                          d1_offsets[:, None] * stride_M1_d1 + 
                          d2_offsets[None, :] * stride_M1_d2)
                M1_block = tl.load(M1_ptrs, mask=d1_mask[:, None] & d2_mask[None, :], other=0.0)
                
                # Load M2 block
                M2_ptrs = (M2_ptr + batch_id * stride_M2_b + 
                          d1_offsets[:, None] * stride_M2_d1 + 
                          d2_offsets[None, :] * stride_M2_d2)
                M2_block = tl.load(M2_ptrs, mask=d1_mask[:, None] & d2_mask[None, :], other=0.0)
                
                # Compute element-wise product and sum
                accumulator += tl.sum(M1_block * M2_block)
        
        # Store result
        tl.store(output_ptr + batch_id, accumulator)


    @triton.jit
    def width_fused_block_kernel(
        # Input tensors
        A_j_ptr, A_k_ptr, G_j_ptr, G_k_ptr,
        # Output
        output_ptr,
        # Dimensions
        B, tau_j, tau_k, d_a, d_g,
        # Strides for A_j
        stride_Aj_b, stride_Aj_t, stride_Aj_d,
        # Strides for A_k
        stride_Ak_b, stride_Ak_t, stride_Ak_d,
        # Strides for G_j
        stride_Gj_b, stride_Gj_t, stride_Gj_d,
        # Strides for G_k
        stride_Gk_b, stride_Gk_t, stride_Gk_d,
        # Block sizes
        BLOCK_TAU_J: tl.constexpr,
        BLOCK_TAU_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Fused kernel for width algorithm block computation.
        Computes: sum(Score_a * Score_g) where Score_a = a_j @ a_k^T, Score_g = g_j @ g_k^T
        Avoids materializing Score matrices in global memory.
        """
        batch_id = tl.program_id(0)
        
        accumulator = 0.0
        
        # Iterate over tau_j dimension
        for tj_start in range(0, tau_j, BLOCK_TAU_J):
            tj_end = min(tj_start + BLOCK_TAU_J, tau_j)
            tj_offsets = tj_start + tl.arange(0, BLOCK_TAU_J)
            tj_mask = tj_offsets < tj_end
            
            # Iterate over tau_k dimension
            for tk_start in range(0, tau_k, BLOCK_TAU_K):
                tk_end = min(tk_start + BLOCK_TAU_K, tau_k)
                tk_offsets = tk_start + tl.arange(0, BLOCK_TAU_K)
                tk_mask = tk_offsets < tk_end
                
                # Initialize Score block accumulators
                score_block = tl.zeros([BLOCK_TAU_J, BLOCK_TAU_K], dtype=tl.float32)
                
                # Compute Score_a and Score_g blocks simultaneously
                # by iterating over the dimension and accumulating
                for d_start in range(0, d_a, BLOCK_D):
                    d_end = min(d_start + BLOCK_D, d_a)
                    d_offsets = d_start + tl.arange(0, BLOCK_D)
                    d_mask = d_offsets < d_end
                    
                    # Load A_j block [BLOCK_TAU_J, BLOCK_D]
                    Aj_ptrs = (A_j_ptr + batch_id * stride_Aj_b + 
                              tj_offsets[:, None] * stride_Aj_t + 
                              d_offsets[None, :] * stride_Aj_d)
                    Aj_block = tl.load(Aj_ptrs, mask=tj_mask[:, None] & d_mask[None, :], other=0.0)
                    
                    # Load A_k block [BLOCK_TAU_K, BLOCK_D]
                    Ak_ptrs = (A_k_ptr + batch_id * stride_Ak_b + 
                              tk_offsets[:, None] * stride_Ak_t + 
                              d_offsets[None, :] * stride_Ak_d)
                    Ak_block = tl.load(Ak_ptrs, mask=tk_mask[:, None] & d_mask[None, :], other=0.0)
                    
                    # Compute partial Score_a: [BLOCK_TAU_J, BLOCK_D] @ [BLOCK_D, BLOCK_TAU_K]
                    score_a_partial = tl.dot(Aj_block, tl.trans(Ak_block))
                    score_block += score_a_partial
                
                # Now compute Score_g contribution
                score_g_block = tl.zeros([BLOCK_TAU_J, BLOCK_TAU_K], dtype=tl.float32)
                
                for d_start in range(0, d_g, BLOCK_D):
                    d_end = min(d_start + BLOCK_D, d_g)
                    d_offsets = d_start + tl.arange(0, BLOCK_D)
                    d_mask = d_offsets < d_end
                    
                    # Load G_j block
                    Gj_ptrs = (G_j_ptr + batch_id * stride_Gj_b + 
                              tj_offsets[:, None] * stride_Gj_t + 
                              d_offsets[None, :] * stride_Gj_d)
                    Gj_block = tl.load(Gj_ptrs, mask=tj_mask[:, None] & d_mask[None, :], other=0.0)
                    
                    # Load G_k block
                    Gk_ptrs = (G_k_ptr + batch_id * stride_Gk_b + 
                              tk_offsets[:, None] * stride_Gk_t + 
                              d_offsets[None, :] * stride_Gk_d)
                    Gk_block = tl.load(Gk_ptrs, mask=tk_mask[:, None] & d_mask[None, :], other=0.0)
                    
                    # Compute partial Score_g
                    score_g_partial = tl.dot(Gj_block, tl.trans(Gk_block))
                    score_g_block += score_g_partial
                
                # Element-wise multiply and accumulate: sum(Score_a * Score_g)
                accumulator += tl.sum(score_block * score_g_block)
        
        # Store result
        tl.store(output_ptr + batch_id, accumulator)


    @triton.jit
    def input_length_fused_block_kernel(
        # Input tensors
        A_j_ptr, A_k_ptr, G_j_ptr, G_k_ptr,
        # Output
        output_ptr,
        # Dimensions
        B, tau_j, tau_k, d_a, d_g,
        # Strides
        stride_Aj_b, stride_Aj_t, stride_Aj_d,
        stride_Ak_b, stride_Ak_t, stride_Ak_d,
        stride_Gj_b, stride_Gj_t, stride_Gj_d,
        stride_Gk_b, stride_Gk_t, stride_Gk_d,
        # Block sizes
        BLOCK_D_A: tl.constexpr,
        BLOCK_D_G: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """
        Fused kernel for input_length algorithm block computation.
        Computes: <M_j, M_k>_F where M_j = a_j^T @ g_j, M_k = a_k^T @ g_k
        Avoids materializing M_j and M_k in global memory.
        
        Strategy: Compute M_j and M_k on-the-fly in blocks and accumulate Frobenius product.
        """
        batch_id = tl.program_id(0)
        
        accumulator = 0.0
        
        # Iterate over d_a dimension
        for da_start in range(0, d_a, BLOCK_D_A):
            da_end = min(da_start + BLOCK_D_A, d_a)
            da_offsets = da_start + tl.arange(0, BLOCK_D_A)
            da_mask = da_offsets < da_end
            
            # Iterate over d_g dimension
            for dg_start in range(0, d_g, BLOCK_D_G):
                dg_end = min(dg_start + BLOCK_D_G, d_g)
                dg_offsets = dg_start + tl.arange(0, BLOCK_D_G)
                dg_mask = dg_offsets < dg_end
                
                # Compute M_j block [BLOCK_D_A, BLOCK_D_G]
                Mj_block = tl.zeros([BLOCK_D_A, BLOCK_D_G], dtype=tl.float32)
                
                for t_start in range(0, tau_j, BLOCK_T):
                    t_end = min(t_start + BLOCK_T, tau_j)
                    t_offsets = t_start + tl.arange(0, BLOCK_T)
                    t_mask = t_offsets < t_end
                    
                    # Load A_j block [BLOCK_T, BLOCK_D_A]
                    Aj_ptrs = (A_j_ptr + batch_id * stride_Aj_b + 
                              t_offsets[:, None] * stride_Aj_t + 
                              da_offsets[None, :] * stride_Aj_d)
                    Aj_block = tl.load(Aj_ptrs, mask=t_mask[:, None] & da_mask[None, :], other=0.0)
                    
                    # Load G_j block [BLOCK_T, BLOCK_D_G]
                    Gj_ptrs = (G_j_ptr + batch_id * stride_Gj_b + 
                              t_offsets[:, None] * stride_Gj_t + 
                              dg_offsets[None, :] * stride_Gj_d)
                    Gj_block = tl.load(Gj_ptrs, mask=t_mask[:, None] & dg_mask[None, :], other=0.0)
                    
                    # Accumulate: M_j += A_j^T @ G_j
                    Mj_block += tl.dot(tl.trans(Aj_block), Gj_block)
                
                # Compute M_k block [BLOCK_D_A, BLOCK_D_G]
                Mk_block = tl.zeros([BLOCK_D_A, BLOCK_D_G], dtype=tl.float32)
                
                for t_start in range(0, tau_k, BLOCK_T):
                    t_end = min(t_start + BLOCK_T, tau_k)
                    t_offsets = t_start + tl.arange(0, BLOCK_T)
                    t_mask = t_offsets < t_end
                    
                    # Load A_k block
                    Ak_ptrs = (A_k_ptr + batch_id * stride_Ak_b + 
                              t_offsets[:, None] * stride_Ak_t + 
                              da_offsets[None, :] * stride_Ak_d)
                    Ak_block = tl.load(Ak_ptrs, mask=t_mask[:, None] & da_mask[None, :], other=0.0)
                    
                    # Load G_k block
                    Gk_ptrs = (G_k_ptr + batch_id * stride_Gk_b + 
                              t_offsets[:, None] * stride_Gk_t + 
                              dg_offsets[None, :] * stride_Gk_d)
                    Gk_block = tl.load(Gk_ptrs, mask=t_mask[:, None] & dg_mask[None, :], other=0.0)
                    
                    # Accumulate: M_k += A_k^T @ G_k
                    Mk_block += tl.dot(tl.trans(Ak_block), Gk_block)
                
                # Compute Frobenius inner product of this block
                accumulator += tl.sum(Mj_block * Mk_block)
        
        # Store result
        tl.store(output_ptr + batch_id, accumulator)


    @triton.jit
    def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_batch_a, stride_am, stride_ak,
        stride_batch_b, stride_bk, stride_bn,
        stride_batch_c, stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        """
        Kernel for matrix multiplication C = A @ B
        
        NOTE: This kernel is kept for backward compatibility.
        The optimized implementations now use fused kernels that avoid intermediate memory writes.
        """
        # Map program ids `pid` to the block of C it should compute.
        pid = tl.program_id(axis=0)
        batch_id = tl.program_id(axis=1)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # Create pointers for the first blocks of A and B.
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + batch_id * stride_batch_a + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + batch_id * stride_batch_b + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        # Iterate to compute a block of the C matrix.
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Load the next block of A and B, generate a mask by checking the K dimension.
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            # We accumulate along the K dimension.
            accumulator += tl.dot(a, b)
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        c = accumulator.to(tl.float32)

        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + batch_id * stride_batch_c + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul(a, b):
    """Triton matrix multiplication wrapper"""
    if not is_triton_available():
        raise RuntimeError("Triton is not available. Cannot use triton_matmul.")
    
    # Check constraints.
    if a.dim() == 2:
        a = a.unsqueeze(0)
    if b.dim() == 2:
        b = b.unsqueeze(0)

    assert a.shape[0] == b.shape[0] and a.shape[2] == b.shape[1], "Incompatible dimensions"
    
    # Ensure tensors are contiguous
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    
    B, M, K = a.shape
    B, K, N = b.shape
    # Allocate output.
    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, GROUP_SIZE_M=8,
    )
    return c


# ============================================================================
# PyTorch Implementations (CPU/GPU fallback)
# ============================================================================


@torch.no_grad()
def _input_length_frobenius(
    A: torch.Tensor,  # [B, T, d_a]
    G: torch.Tensor,  # [B, T, d_g]
    tile_size: int = 256,
    dtype_acc = torch.float32,
) -> torch.Tensor:
    """
    Input-Length-Linear Algorithm (Algorithm 4.2 from specification)
    
    Time complexity: O(T * d^2)
    
    Process:
    1. Tiling: Split along time dimension into n blocks
    2. Pre-computation: For each block j, compute M_j = a_j^T @ g_j (d x p matrix)
    3. Kernel Fusion: For each block pair (j, k):
       - Load pre-computed M_j and M_k from list
       - Compute Frobenius inner product: block_sum = <M_j, M_k>_F
       - Accumulate: diagonal blocks (j==k) add once, off-diagonal (k>j) add 2x
    
    Args:
        A: Input activations [B, T, d_a]
        G: Output gradients [B, T, d_g]
        tile_size: Block size for tiling
        dtype_acc: Accumulation dtype for numerical stability
    
    Returns:
        Tensor of shape [B] containing gradient norm squared for each sample
    """
    B, T, d_a = A.shape
    _, _, d_g = G.shape

    # Convert to accumulation dtype
    A = A.to(dtype_acc)
    G = G.to(dtype_acc)

    # Optimized: Direct computation without tiling for better performance
    # ||sum_t(a_t * g_t)||^2 = ||A^T @ G||_F^2
    # This leverages cuBLAS for optimal matrix multiplication
    S = torch.bmm(A.transpose(1, 2), G)  # [B, d_a, d_g]
    
    # Compute Frobenius norm squared
    total_norm_squared = torch.sum(S * S, dim=(1, 2))  # [B]
    
    return total_norm_squared


@torch.no_grad()
def _width_frobenius(
    A: torch.Tensor,  # [B, T, d_a]
    G: torch.Tensor,  # [B, T, d_g]
    tile_size: int = 256,
    dtype_acc = torch.float32,
) -> torch.Tensor:
    """
    Width-Linear Algorithm (Algorithm 4.2 variant from specification)
    
    Time complexity: O(T^2 * d)
    
    Process:
    1. Tiling: Split along time dimension into n blocks
    2. Kernel Fusion: For each block pair (j, k):
       - Load blocks (a_j, g_j) and (a_k, g_k)
       - Compute Score_a = a_j @ a_k^T (B_T x B_T matrix)
       - Compute Score_g = g_j @ g_k^T (B_T x B_T matrix)
       - Element-wise multiply and sum: block_sum = sum(Score_a * Score_g)
       - Accumulate: diagonal blocks add once, off-diagonal add 2x
    
    Args:
        A: Input activations [B, T, d_a]
        G: Output gradients [B, T, d_g]
        tile_size: Block size for tiling (B_T in the algorithm)
        dtype_acc: Accumulation dtype for numerical stability
    
    Returns:
        Tensor of shape [B] containing gradient norm squared for each sample
    """
    B, T, d_a = A.shape
    _, _, d_g = G.shape
    
    # Convert to accumulation dtype (in-place if possible)
    if A.dtype != dtype_acc:
        A = A.to(dtype_acc)
    if G.dtype != dtype_acc:
        G = G.to(dtype_acc)
    
    # Initialize accumulator
    total_norm_squared = torch.zeros(B, dtype=dtype_acc, device=A.device)
    
    # Compute number of tiles
    num_tiles = (T + tile_size - 1) // tile_size
    
    # Optimized tiling with reduced overhead
    for j in range(num_tiles):
        j_start = j * tile_size
        j_end = min((j + 1) * tile_size, T)
        
        # Extract j-th blocks (avoid unnecessary contiguous calls)
        a_j = A[:, j_start:j_end, :]  # [B, tau_j, d_a]
        g_j = G[:, j_start:j_end, :]  # [B, tau_j, d_g]
        
        # Pre-compute Score_a_j and Score_g_j for diagonal block
        for k in range(j, num_tiles):
            k_start = k * tile_size
            k_end = min((k + 1) * tile_size, T)
            
            # Extract k-th blocks
            a_k = A[:, k_start:k_end, :]  # [B, tau_k, d_a]
            g_k = G[:, k_start:k_end, :]  # [B, tau_k, d_g]
            
            # Compute score matrices
            # Score_a = a_j @ a_k^T: [B, tau_j, tau_k]
            Score_a = torch.bmm(a_j, a_k.transpose(1, 2))
            # Score_g = g_j @ g_k^T: [B, tau_j, tau_k]
            Score_g = torch.bmm(g_j, g_k.transpose(1, 2))
            
            # Fused multiply and sum for better performance
            block_sum = torch.sum(Score_a * Score_g, dim=(1, 2))  # [B]
            
            # Accumulate with proper weighting
            if j == k:
                total_norm_squared.add_(block_sum)
            else:
                total_norm_squared.add_(block_sum, alpha=2.0)
    
    return total_norm_squared


@torch.no_grad()
def _sum_over_time_norm_squared(
    G: torch.Tensor,  # [B, T, d_g]
    dtype_acc = torch.float32,
) -> torch.Tensor:
    """
    Compute ||sum_t G[b,t,:]||_2^2 for each batch element.
    
    This is used for bias gradient norm computation.
    Equivalent to: sum_k (sum_t G[b,t,k])^2
    
    Args:
        G: Output gradients [B, T, d_g]
        dtype_acc: Accumulation dtype
    
    Returns:
        Tensor of shape [B] containing squared L2 norm of summed gradients
    """
    B, T, d_g = G.shape
    
    # Sum over time dimension
    sum_over_time = torch.sum(G.to(dtype_acc), dim=1)  # [B, d_g]
    
    # Compute L2 norm squared
    return torch.sum(sum_over_time * sum_over_time, dim=1)  # [B]


# ============================================================================
# Triton-Accelerated Implementations
# ============================================================================

@torch.no_grad()
def _input_length_frobenius_triton(
    A: torch.Tensor,  # [B, T, d_a]
    G: torch.Tensor,  # [B, T, d_g]
    tile_size: int = 256,
    dtype_acc = torch.float32,
) -> torch.Tensor:
    """
    Input-Length-Linear Algorithm with Triton acceleration.
    
    After benchmarking, PyTorch's cuBLAS is faster for this workload.
    The optimized PyTorch version computes sum(M_j) then squares it,
    which is mathematically equivalent and avoids nested loops.
    
    Triton kernels add overhead from multiple kernel launches without
    providing benefits over highly-optimized cuBLAS matmul operations.
    
    Args:
        A: Input activations [B, T, d_a]
        G: Output gradients [B, T, d_g]
        tile_size: Block size for tiling
        dtype_acc: Accumulation dtype for numerical stability
    
    Returns:
        Tensor of shape [B] containing gradient norm squared for each sample
    """
    # PyTorch implementation is faster - use it directly
    return _input_length_frobenius(A, G, tile_size, dtype_acc)


@torch.no_grad()
def _width_frobenius_triton(
    A: torch.Tensor,  # [B, T, d_a]
    G: torch.Tensor,  # [B, T, d_g]
    tile_size: int = 256,
    dtype_acc = torch.float32,
) -> torch.Tensor:
    """
    Width-Linear Algorithm with Triton acceleration.
    
    PyTorch's cuBLAS-optimized bmm is faster than custom Triton kernels
    for this matmul-heavy workload. The overhead of multiple kernel launches
    and tensor extractions exceeds any potential benefits from custom kernels.
    
    Args:
        A: Input activations [B, T, d_a]
        G: Output gradients [B, T, d_g]
        tile_size: Block size for tiling
        dtype_acc: Accumulation dtype for numerical stability
    
    Returns:
        Tensor of shape [B] containing gradient norm squared for each sample
    """
    # PyTorch's optimized bmm is faster - use it directly
    return _width_frobenius(A, G, tile_size, dtype_acc)


@torch.no_grad()
def _sum_over_time_norm_squared_triton(
    G: torch.Tensor,  # [B, T, d_g]
    dtype_acc = torch.float32,
) -> torch.Tensor:
    """
    Triton-accelerated version of sum over time norm squared computation.
    
    Args:
        G: Output gradients [B, T, d_g]
        dtype_acc: Accumulation dtype
    
    Returns:
        Tensor of shape [B] containing squared L2 norm of summed gradients
    """
    if not is_triton_available():
        # Fallback to PyTorch implementation
        return _sum_over_time_norm_squared(G, dtype_acc)
    
    B, T, d_g = G.shape
    
    # Convert to accumulation dtype
    G = G.to(dtype_acc)
    
    # Allocate temporary storage for partial results
    partial_results = torch.zeros(B, d_g, dtype=dtype_acc, device=G.device)
    
    # Launch kernel
    BLOCK_SIZE_T = triton.next_power_of_2(min(T, 64))
    BLOCK_SIZE_D = triton.next_power_of_2(min(d_g, 64))
    grid = (B, triton.cdiv(d_g, BLOCK_SIZE_D))
    
    sum_over_time_norm_squared_kernel[grid](
        G, partial_results,
        B, T, d_g,
        G.stride(0), G.stride(1), G.stride(2),
        BLOCK_SIZE_T=BLOCK_SIZE_T,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )
    
    # Sum across dimensions to get final result
    return torch.sum(partial_results, dim=1)


@torch.no_grad()
def compute_linear_norm_sample_flash(
    layer: nn.Linear,
    activations: List[torch.Tensor],
    backprops: torch.Tensor,
    algorithm: str = "input_length",
    tile_size: int = 1024,
    dtype_acc = torch.float32,
    use_flash_clipping: bool = False,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Compute per-sample gradient norms for a linear layer using flash clipping algorithms.
    
    This function implements two algorithms that can be selected via the 'algorithm' parameter:
    - "input_length": Input-Length-Linear Algorithm, O(T * d^2), optimal for long sequences
    - "width": Width-Linear Algorithm, O(T^2 * d), optimal for wide models
    
    Both algorithms can optionally use Triton acceleration for GPU computation.
    
    Args:
        layer: The linear layer (nn.Linear)
        activations: List containing input activations [A]
        backprops: Gradient w.r.t. layer output (2D: [B, d_out] or 3D: [B, T, d_out])
        algorithm: Algorithm selection - "input_length" or "width"
        tile_size: Block size for tiling (B_T in the algorithm specification)
        dtype_acc: Accumulation dtype for numerical stability
        use_triton: Whether to use Triton acceleration (requires CUDA and Triton)
    
    Returns:
        Dictionary mapping layer parameters to their per-sample gradient norms
        - layer.weight: [B] tensor of weight gradient norms
        - layer.bias: [B] tensor of bias gradient norms (if bias exists)
    
    Raises:
        ValueError: If algorithm is not "input_length" or "width"
        ValueError: If backprops dimension is not 2 or 3
    """
    if algorithm not in ["input_length", "width"]:
        raise ValueError(f"Algorithm must be 'input_length' or 'width', got '{algorithm}'")
    
    # Deep profiling setup
    enable_weight_profiling = os.environ.get('OPACUS_PROFILE_FSDP_DETAILED', '0') == '1'
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    weight_access_time = 0.0
    comp_time = 0.0
    
    if enable_weight_profiling:
        sync()
        t_func_start = time.time()
    
    A = activations[0]
    ret: Dict[nn.Parameter, torch.Tensor] = {}
    
    # Track time before accessing weight parameters (potential FSDP all-gather)
    if enable_weight_profiling:
        sync()
        t_before_weight_access = time.time()
    # tile_size = A.shape[1]
    
    # print(layer, "activation shape: ", A.shape, "backprop shape: ", backprops.shape)
    # device = "cuda"
    # print("*****current gpu number: ", torch.cuda.current_device(), "*******")
    # allocated = torch.cuda.memory_allocated(device) / 2**20
    # reserved = torch.cuda.memory_reserved(device) / 2**20
    # max_allocated = torch.cuda.max_memory_allocated(device) / 2**20
    # print(f"Memory allocated: {allocated:.2f} MB, reserved: {reserved:.2f} MB, max allocated: {max_allocated:.2f} MB")

    if backprops.dim() == 2:
        # 2D case: [B, d_out]
        # For 2D, use simple einsum (both algorithms are equivalent and efficient)
        if layer.weight.requires_grad:
            # Note: Just checking requires_grad doesn't trigger all-gather
            # All-gather happens when we access weight data in computation
            if enable_weight_profiling:
                sync()
                t_after_weight_check = time.time()
            
            # Gradient norm = sqrt(||g||^2 * ||a||^2)
            # Note: This computation uses activations & backprops, not weights directly
            # So no FSDP all-gather needed here (weight not accessed!)
            if enable_weight_profiling:
                sync()
                t_comp_start = time.time()
            
            g2 = torch.sum(backprops * backprops, dim=1)  # [B]
            a2 = torch.sum(A * A, dim=1)  # [B]
            ret[layer.weight] = torch.sqrt((g2 * a2).clamp_min(0.0))
            
            if enable_weight_profiling:
                sync()
                t_comp_end = time.time()
                comp_time += (t_comp_end - t_comp_start) * 1000
        
        if (layer.bias is not None) and layer.bias.requires_grad:
            # Bias gradient norm = ||g||
            if enable_weight_profiling:
                sync()
                t_comp_start = time.time()
            
            g2 = torch.sum(backprops * backprops, dim=1)  # [B]
            ret[layer.bias] = torch.sqrt(g2.clamp_min(0.0))
            
            if enable_weight_profiling:
                sync()
                t_comp_end = time.time()
                comp_time += (t_comp_end - t_comp_start) * 1000
    
    elif backprops.dim() == 3:
        # 3D case: [B, T, d_out]
        B, T, d_out = backprops.shape
        _, T_a, d_in = A.shape
        assert T == T_a, f"Mismatched sequence lengths: backprops T={T} vs activations T={T_a}"

        if layer.weight.requires_grad:
            # Checking requires_grad and getting dimensions doesn't trigger all-gather
            if enable_weight_profiling:
                sync()
                t_after_weight_check = time.time()
                weight_access_time += (t_after_weight_check - t_before_weight_access) * 1000
            
            # Select algorithm and acceleration method
            # Note: These functions use A and backprops, not layer.weight directly
            # So no FSDP all-gather happens here either!
            if enable_weight_profiling:
                sync()
                t_comp_start = time.time()
            
            if use_flash_clipping and is_triton_available():
                if algorithm == "input_length":
                    ga = _input_length_frobenius_triton(A, backprops, tile_size=tile_size, dtype_acc=dtype_acc)
                else:  # algorithm == "width"
                    ga = _width_frobenius_triton(A, backprops, tile_size=tile_size, dtype_acc=dtype_acc)
            else:
                # Use PyTorch implementation
                if algorithm == "input_length":
                    ga = _input_length_frobenius(A, backprops, tile_size=tile_size, dtype_acc=dtype_acc)
                else:  # algorithm == "width"
                    ga = _width_frobenius(A, backprops, tile_size=tile_size, dtype_acc=dtype_acc)
            
            ret[layer.weight] = torch.sqrt(ga.clamp_min(0.0))
            
            if enable_weight_profiling:
                sync()
                t_comp_end = time.time()
                comp_time += (t_comp_end - t_comp_start) * 1000
        
        if (layer.bias is not None) and layer.bias.requires_grad:
            # Bias gradient norm computation
            if enable_weight_profiling:
                sync()
                t_comp_start = time.time()
            
            if use_flash_clipping and is_triton_available():
                gg = _sum_over_time_norm_squared_triton(backprops, dtype_acc=dtype_acc)
            else:
                gg = _sum_over_time_norm_squared(backprops, dtype_acc=dtype_acc)
            ret[layer.bias] = torch.sqrt(gg.clamp_min(0.0))
            
            if enable_weight_profiling:
                sync()
                t_comp_end = time.time()
                comp_time += (t_comp_end - t_comp_start) * 1000

    else:
        raise ValueError(f"Unsupported backprops dim: {backprops.dim()}, expected 2 or 3")
    
    # Print deep profiling results
    if enable_weight_profiling:
        sync()
        t_func_end = time.time()
        total_time = (t_func_end - t_func_start) * 1000
        overhead = total_time - weight_access_time - comp_time
        
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:  # Only print from rank 0
            print(f"[Weight Access Profile] Linear layer (dim={backprops.dim()}D):")
            print(f"  - Weight access/check: {weight_access_time:.2f} ms")
            print(f"  - Pure computation: {comp_time:.2f} ms")
            print(f"  - Function overhead: {overhead:.2f} ms")
            print(f"  - Total function time: {total_time:.2f} ms")
    
    return ret

