"""
Flash Clipping Algorithms for Linear Layers - Async Stream Safe Version

This module implements async CUDA stream-safe versions of flash clipping algorithms:
1. Input-Length-Linear Algorithm: O(T * d^2) - optimal for long sequences
2. Width-Linear Algorithm: O(T^2 * d) - optimal for wide models

Key difference from triton_kernels.py:
- Parameter-safe interface that avoids accessing layer.weight/layer.bias inside streams
- Designed to be called from async CUDA streams without triggering FSDP synchronization
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn

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
    
    # Step 1: Transpose
    A_t = A.transpose(1, 2)
    
    # Step 2: Batch matrix multiply
    S = torch.bmm(A_t, G)  # [B, d_a, d_g]
    
    # Step 3: Element-wise square
    S_squared = S * S
    
    # Step 4: Reduction
    total_norm_squared = torch.sum(S_squared, dim=(1, 2))  # [B]
    
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
def compute_linear_norm_sample_flash_async(
    activations: List[torch.Tensor],
    backprops: torch.Tensor,
    weight_requires_grad: bool,
    bias_requires_grad: bool,
    weight_param: Optional[nn.Parameter] = None,
    bias_param: Optional[nn.Parameter] = None,
    algorithm: str = "input_length",
    tile_size: int = 1024,
    dtype_acc = torch.float32,
    use_flash_clipping: bool = False,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Async CUDA stream-safe version of compute_linear_norm_sample_flash.
    
    This function is designed to be called from async CUDA streams without
    triggering FSDP parameter gathering or synchronization. It avoids
    accessing any layer parameter attributes inside the computation.
    
    Key differences from the original:
    - Takes boolean flags instead of checking layer.weight.requires_grad
    - Takes parameter references as arguments instead of accessing layer.weight/bias
    - No parameter attribute access inside computation blocks
    
    Args:
        activations: List containing input activations [A]
        backprops: Gradient w.r.t. layer output (2D: [B, d_out] or 3D: [B, T, d_out])
        weight_requires_grad: Whether weight requires gradient (checked in main thread)
        bias_requires_grad: Whether bias requires gradient (checked in main thread)
        weight_param: Weight parameter reference for dict key (optional)
        bias_param: Bias parameter reference for dict key (optional)
        algorithm: Algorithm selection - "input_length" or "width"
        tile_size: Block size for tiling (B_T in the algorithm specification)
        dtype_acc: Accumulation dtype for numerical stability
        use_flash_clipping: Whether flash clipping is enabled (for compatibility)
    
    Returns:
        Dictionary mapping parameters to their per-sample gradient norms
        - weight_param: [B] tensor of weight gradient norms (if weight_requires_grad)
        - bias_param: [B] tensor of bias gradient norms (if bias_requires_grad)
    
    Raises:
        ValueError: If algorithm is not "input_length" or "width"
        ValueError: If backprops dimension is not 2 or 3
    """
    if algorithm not in ["input_length", "width"]:
        raise ValueError(f"Algorithm must be 'input_length' or 'width', got '{algorithm}'")
    
    A = activations[0]
    ret: Dict[nn.Parameter, torch.Tensor] = {}
    
    if backprops.dim() == 2:
        # 2D case: [B, d_out]
        # For 2D, use simple einsum (both algorithms are equivalent and efficient)
        if weight_requires_grad:
            # Gradient norm = sqrt(||g||^2 * ||a||^2)
            g2 = torch.sum(backprops * backprops, dim=1)  # [B]
            a2 = torch.sum(A * A, dim=1)  # [B]
            norm_weight = torch.sqrt((g2 * a2).clamp_min(0.0))
            if weight_param is not None:
                ret[weight_param] = norm_weight
            else:
                # Fallback: return with a placeholder key
                ret['weight'] = norm_weight
        
        if bias_requires_grad:
            # Bias gradient norm = ||g||
            g2 = torch.sum(backprops * backprops, dim=1)  # [B]
            norm_bias = torch.sqrt(g2.clamp_min(0.0))
            if bias_param is not None:
                ret[bias_param] = norm_bias
            else:
                ret['bias'] = norm_bias
    
    elif backprops.dim() == 3:
        # 3D case: [B, T, d_out]
        B, T, d_out = backprops.shape
        _, T_a, d_in = A.shape
        assert T == T_a, f"Mismatched sequence lengths: backprops T={T} vs activations T={T_a}"

        if weight_requires_grad:
            # Select algorithm - use PyTorch implementation
            if algorithm == "input_length":
                ga = _input_length_frobenius(A, backprops, tile_size=tile_size, dtype_acc=dtype_acc)
            else:  # algorithm == "width"
                ga = _width_frobenius(A, backprops, tile_size=tile_size, dtype_acc=dtype_acc)
        
            norm_weight = torch.sqrt(ga.clamp_min(0.0))
            if weight_param is not None:
                ret[weight_param] = norm_weight
            else:
                ret['weight'] = norm_weight
        
        if bias_requires_grad:
            # Bias gradient norm computation
            gg = _sum_over_time_norm_squared(backprops, dtype_acc=dtype_acc)
            norm_bias = torch.sqrt(gg.clamp_min(0.0))
            if bias_param is not None:
                ret[bias_param] = norm_bias
            else:
                ret['bias'] = norm_bias

    else:
        raise ValueError(f"Unsupported backprops dim: {backprops.dim()}, expected 2 or 3")
    
    return ret

