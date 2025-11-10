from typing import Dict, List
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl
from opt_einsum import contract


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
    """Kernel for matrix multiplication C = A @ B"""
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





@torch.no_grad()
def _triton_frobenius_inner_over_T(
    A: torch.Tensor,  # [B, T, d_a]
    G: torch.Tensor,  # [B, T, d_g]
    tile_size: int = 256,
    dtype_acc = torch.float32,
) -> torch.Tensor:
    """
    Optimized version that computes Frobenius inner product <G@G^T, A@A^T>
    using tiling to avoid memory issues with large T.
    """
    B, T, d_a = A.shape
    _, _, d_g = G.shape
    
    # Convert to accumulation dtype
    A = A.to(dtype_acc)
    G = G.to(dtype_acc)
    
    ga = torch.zeros(B, dtype=dtype_acc, device=A.device)
    num_tiles = (T + tile_size - 1) // tile_size

    for p in range(num_tiles):
        ps = p * tile_size
        pe = min((p + 1) * tile_size, T)
        A_p = A[:, ps:pe, :].contiguous()
        G_p = G[:, ps:pe, :].contiguous()

        # Diagonal block (p, p): sum(G_p @ G_p^T * A_p @ A_p^T)
        # Use efficient batched operations
        Sg_pp = triton_matmul(G_p, G_p.transpose(-2, -1))  # [B, tau_p, tau_p]
        Sa_pp = triton_matmul(A_p, A_p.transpose(-2, -1))  # [B, tau_p, tau_p]
        ga += torch.sum(Sg_pp * Sa_pp, dim=(1, 2))

        # Off-diagonal blocks (q < p): 2 * sum(G_p @ G_q^T * A_p @ A_q^T)
        for q in range(p):
            qs = q * tile_size
            qe = min((q + 1) * tile_size, T)
            A_q = A[:, qs:qe, :].contiguous()
            G_q = G[:, qs:qe, :].contiguous()

            Sg_pq = triton_matmul(G_p, G_q.transpose(-2, -1))  # [B, tau_p, tau_q]
            Sa_pq = triton_matmul(A_p, A_q.transpose(-2, -1))  # [B, tau_p, tau_q]
            ga += 2.0 * torch.sum(Sg_pq * Sa_pq, dim=(1, 2))

    return ga


@torch.no_grad()
def _triton_sum_over_time_norm_squared(
    G: torch.Tensor,  # [B, T, d_g]
    dtype_acc = torch.float32,
) -> torch.Tensor:
    """
    Triton-accelerated version of sum over time norm squared computation.
    Computes ||sum_t G[b,t,:]||_2^2 for each batch element.
    """
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
def compute_linear_norm_sample_triton(
    layer: nn.Linear,
    activations: List[torch.Tensor],
    backprops: torch.Tensor,
    tile_size: int = 512,
    dtype_acc = torch.float32,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Triton-accelerated version of compute_linear_norm_sample.
    """
    A = activations[0]
    ret: Dict[nn.Parameter, torch.Tensor] = {}

    if backprops.dim() == 2:
        # For 2D case, use the original implementation as it's already efficient
        if layer.weight.requires_grad:
            g2 = contract('bi,bi->b', backprops, backprops)
            a2 = contract('bi,bi->b', A, A)
            ret[layer.weight] = torch.sqrt((g2 * a2).clamp_min(0.0))

        if (layer.bias is not None) and layer.bias.requires_grad:
            g2 = contract('bi,bi->b', backprops, backprops)
            ret[layer.bias] = torch.sqrt(g2.clamp_min(0.0))

    elif backprops.dim() == 3:
        B, T, d_out = backprops.shape
        _, T_a, d_in = A.shape
        assert T == T_a, f"Mismatched sequence lengths: backprops T={T} vs activations T={T_a}"

        if layer.weight.requires_grad:
            ga = _triton_frobenius_inner_over_T(A, backprops, tile_size=tile_size, dtype_acc=dtype_acc)
            ret[layer.weight] = torch.sqrt(ga.clamp_min(0.0))

        if (layer.bias is not None) and layer.bias.requires_grad:
            gg = _triton_sum_over_time_norm_squared(backprops, dtype_acc=dtype_acc)
            ret[layer.bias] = torch.sqrt(gg.clamp_min(0.0))

    else:
        raise ValueError(f"Unsupported backprops dim: {backprops.dim()}")

    return ret