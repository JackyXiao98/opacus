import torch
import triton
import triton.language as tl

# ============================================================================
# Triton Kernel: Fused Weight Gradient & Flash Norm (Optimized)
# 
# Function:
# 1. Computes dW = sum_b( G[b]^T @ X[b] )  -> Standard Weight Gradient
# 2. Computes Norms[b] = || G[b]^T @ X[b] ||_F^2  -> Per-sample Gradient Norm
# 
# Optimization Update:
# - Added L2 Cache Swizzling (Grouped Program ID) to match CuBLAS performance.
# - Tuned Block Sizes for better occupancy on modern GPUs (A100/H100/3090).
# ============================================================================

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
    BLOCK_K: tl.constexpr,  # Tile size for T
    GROUP_SIZE_M: tl.constexpr, # Group size for L2 Swizzling
):
    # --- L2 Cache Swizzling Logic ---
    # Map 1D PID to 2D Tiled PID with locality awareness
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(Din, BLOCK_M)
    num_pid_n = tl.cdiv(Dout, BLOCK_N)
    
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    # Re-mapped PIDs
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # --------------------------------
    
    # 1. Prepare offsets for output matrix tiles
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M) # Din range
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N) # Dout range
    
    # Mask for boundary checks
    mask_m = offs_m < Din
    mask_n = offs_n < Dout
    
    # 2. Initialize Accumulator for GLOBAL Gradient
    acc_global = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    
    # 3. Outer Loop: Iterate over Batch dimension
    for b in range(B):
        
        # Initialize Accumulator for PER-SAMPLE Gradient (dW_b)
        acc_b = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        
        # Pointers to current batch's X and G
        x_base = X_ptr + b * stride_x_b
        g_base = G_ptr + b * stride_g_b
        
        # 4. Inner Loop: Iterate over Sequence dimension
        for k in range(0, T, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)
            mask_k = offs_k < T
            
            # Load X tile (T, Din) - Transposed access needed for logic? 
            # Logic: dW = G.T @ X.
            # G [T, Dout], X [T, Din].
            # Triton dot(A, B, trans_a=True) computes A.T @ B.
            # We load G as A (T, Dout), X as B (T, Din).
            
            # Load X tile: [BLOCK_K, BLOCK_M]
            x_ptrs = x_base + (offs_k[:, None] * stride_x_t + offs_m[None, :] * stride_x_d)
            x_tile = tl.load(x_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
            
            # Load G tile: [BLOCK_K, BLOCK_N]
            g_ptrs = g_base + (offs_k[:, None] * stride_g_t + offs_n[None, :] * stride_g_d)
            g_tile = tl.load(g_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            
            # Compute: G_tile.T @ X_tile -> [BLOCK_N, BLOCK_K] @ [BLOCK_K, BLOCK_M] -> [BLOCK_N, BLOCK_M]
            acc_b = tl.dot(g_tile, x_tile, acc_b, trans_a=True)
            
        # --- Per-Batch Processing ---
        
        # A. Accumulate to Global Gradient
        acc_global += acc_b
        
        # B. Compute Norm Contribution
        mask_tile = mask_n[:, None] & mask_m[None, :]
        
        # Optimization: Pre-square in registers before applying mask
        # This keeps the pipeline busy
        acc_b_sq = acc_b * acc_b
        valid_acc_b_sq = tl.where(mask_tile, acc_b_sq, 0.0)
        
        norm_tile = tl.sum(valid_acc_b_sq)
        
        # Atomic Add to Norms[b]
        # Since grid size is usually < 1000 blocks, collision is low.
        norm_ptr = Norms_ptr + b * stride_norms_b
        tl.atomic_add(norm_ptr, norm_tile)
        
    # 5. Store Global Gradient
    offs_dw = offs_n[:, None] * stride_dw_out + offs_m[None, :] * stride_dw_in
    tl.store(DW_ptr + offs_dw, acc_global, mask=mask_n[:, None] & mask_m[None, :])


def fused_backward_weight(
    x: torch.Tensor,     # [B, S, Din]
    grad_out: torch.Tensor, # [B, S, Dout]
    norms_buf: torch.Tensor # [B]
) -> torch.Tensor:
    """
    Computes standard grad_weight AND accumulates per-sample norms.
    """
    B, T, Din = x.shape
    _, _, Dout = grad_out.shape
    
    # Output Gradient Weight
    grad_weight = torch.empty((Dout, Din), device=x.device, dtype=torch.float32)
    
    # Tuning config (Aggressive settings for performance)
    # BLOCK_M/N = 128 is usually sweet spot for A100/3090
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32 # Smaller K to increase parallelism inside dot? Or 64.
    GROUP_SIZE_M = 8 # Critical for L2 Cache hit rate
    
    # Calculate grid based on 1D launch for Swizzling
    grid = lambda META: (
        triton.cdiv(Din, META['BLOCK_M']) * triton.cdiv(Dout, META['BLOCK_N']),
    )
    
    _fused_backward_kernel[grid](
        x, grad_out, grad_weight, norms_buf,
        B, T, Din, Dout,
        x.stride(0), x.stride(1), x.stride(2),
        grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
        grad_weight.stride(0), grad_weight.stride(1),
        norms_buf.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=8,  # Increased warps for latency hiding
        num_stages=4  # Increased stages for pipelined loads
    )
    
    return grad_weight

# ============================================================================
# Integration & Benchmark
# ============================================================================

class FusedBackwardLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, norm_buf):
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        ctx.norm_buf = norm_buf
        return torch.nn.functional.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        norm_buf = ctx.norm_buf
        
        # 1. Grad Input (Standard)
        # dL/dx = dL/dy @ W
        grad_x = grad_out @ weight
        
        # 2. Fused Grad Weight + Norm
        # We ensure inputs are contiguous for Triton
        x_c = x if x.is_contiguous() else x.contiguous()
        g_c = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
        
        grad_w = fused_backward_weight(x_c, g_c, norm_buf)
        
        # 3. Bias handling (Optional Fusing, usually cheap enough to keep separate)
        grad_b = None
        if ctx.has_bias:
            # Standard PyTorch Sum
            grad_b = grad_out.sum(dim=(0, 1))
            
            # Bias Norm Contribution to Buffer
            # || sum_t(g_bt) ||^2
            # This is relatively cheap: [B, S, D] -> [B, D] -> [B]
            bias_sums = grad_out.sum(dim=1) 
            bias_norm_sq = bias_sums.pow(2).sum(dim=1)
            norm_buf.add_(bias_norm_sq)
            
        return grad_x, grad_w, grad_b, None

def benchmark_fusion():
    if not torch.cuda.is_available(): return
    device = "cuda"
    torch.manual_seed(0)
    
    B, S, Din, Dout = 4, 1024, 1024, 1024
    
    print(f"Benchmarking Fused Backward: B={B}, S={S}, Din={Din}, Dout={Dout}")
    
    x = torch.randn(B, S, Din, device=device, requires_grad=True)
    w = torch.randn(Dout, Din, device=device, requires_grad=True)
    dy = torch.randn(B, S, Dout, device=device)
    buf = torch.zeros(B, device=device)
    
    # --- 1. PyTorch Standard (Separate) ---
    def run_torch():
        # A. Grad Weight
        # [B*S, Din]^T @ [B*S, Dout]
        x_flat = x.view(-1, Din)
        dy_flat = dy.view(-1, Dout)
        gw = dy_flat.t() @ x_flat
        
        # B. Flash Norm
        # Per-sample gradients... slow path or separate Triton kernel
        # Using simple PyTorch loop for logic simulation (slowest case)
        # In reality one would use the previous separate Triton kernel
        pass
        return gw

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(50): run_torch()
    end.record()
    torch.cuda.synchronize()
    print(f"PyTorch (Grad Only): {start.elapsed_time(end)/50:.3f} ms")
    
    # --- 2. Triton Fused ---
    # Warmup
    fused_backward_weight(x, dy, buf)
    
    start.record()
    for _ in range(50):
        buf.zero_()
        fused_backward_weight(x, dy, buf)
    end.record()
    torch.cuda.synchronize()
    print(f"Triton Fused (Grad + Norm): {start.elapsed_time(end)/50:.3f} ms")
    
    # Correctness
    gw_torch = (dy.view(-1, Dout).t() @ x.view(-1, Din))
    buf.zero_()
    gw_triton = fused_backward_weight(x, dy, buf)
    
    diff = (gw_torch - gw_triton).abs().max()
    print(f"Max Gradient Diff: {diff:.5f}")
    print(f"Norm Buffer Sample 0: {buf[0].item():.4f}")

if __name__ == "__main__":
    benchmark_fusion()