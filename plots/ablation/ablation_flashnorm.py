#!/usr/bin/env python3
"""
Ablation for FlashNorm Triton fused linear backward.

Compares three variants on a single shape (B=1, T=8196, Din=Dout=1024):
1) triton_no_tma   - Triton fused kernel without TMA/block_ptr or Split-T
2) triton_tma      - fused Triton kernel (uses block pointers/TMA path)
3) triton_splitk   - fused Triton Split-T kernel (split_k configurable)

Outputs:
- CSV with mean/std latency and tokens/sec in the same directory.
- PNG bar chart comparing latency.

Usage:
    python plots/ablation/ablation_flashnorm.py \
        --output-dir plots/ablation/out --iters 20 --warmup 5 --split-k 4

Notes:
- Requires CUDA + Triton for Triton variants; baseline runs on CUDA too.
- Split-K is most relevant for large T; default split_k=4 for T=8196.
- TMA benefits show up on H100; other GPUs still exercise the fused path.
"""
import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

import torch
import triton
import triton.language as tl

# Ensure repository root is on sys.path so flashnorm is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from flashnorm.grad_sample import triton_fused_kernel as tkernel
except Exception as exc:  # pragma: no cover - import guard for runtime
    raise RuntimeError(
        "Failed to import flashnorm.grad_sample.triton_fused_kernel. "
        "Run from repository root so flashnorm is importable."
    ) from exc


@dataclass
class BenchmarkResult:
    variant: str
    mean_ms: float
    std_ms: float
    tokens_per_s: float
    use_tma: bool
    split_k: int
    max_abs_err: float
    rel_l2_err: float
    max_mem_mb: float
    max_reserved_mb: float


def _time_fn(fn: Callable[[], Tuple[torch.Tensor, torch.Tensor]], iters: int, warmup: int) -> Tuple[float, float]:
    """Benchmark callable returning (grad_weight, norms_buf) on CUDA."""
    # Warmup to stabilize kernels
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times_ms = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    times_t = torch.tensor(times_ms, device="cpu", dtype=torch.float32)
    return float(times_t.mean()), float(times_t.std(unbiased=False))


def make_inputs(
    device: torch.device,
    B: int,
    T: int,
    Din: int,
    Dout: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn((B, T, Din), device=device, dtype=dtype)
    grad_out = torch.randn((B, T, Dout), device=device, dtype=dtype)
    return x, grad_out


@triton.jit
def _fused_backward_kernel_no_tma(
    X_ptr, G_ptr, DW_ptr, Norms_ptr,
    B, T, Din, Dout,
    stride_x_b, stride_x_t, stride_x_d,
    stride_g_b, stride_g_t, stride_g_d,
    stride_dw_out, stride_dw_in,
    stride_norms_b,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(Din, BLOCK_M)
    num_pid_n = tl.cdiv(Dout, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < Din
    mask_n = offs_n < Dout
    acc_global = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    for b in range(B):
        acc_b = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        for k in range(0, T, BLOCK_K):
            k_range = k + tl.arange(0, BLOCK_K)
            mask_k = k_range < T

            x_ptrs = (
                X_ptr
                + (b * stride_x_b)
                + (k_range[:, None] * stride_x_t)
                + (offs_m[None, :] * stride_x_d)
            )
            g_ptrs = (
                G_ptr
                + (b * stride_g_b)
                + (k_range[:, None] * stride_g_t)
                + (offs_n[None, :] * stride_g_d)
            )
            x = tl.load(x_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
            g = tl.load(g_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc_b += tl.dot(tl.trans(g), x, out_dtype=tl.float32)

        acc_global += acc_b
        mask_tile = mask_n[:, None] & mask_m[None, :]
        acc_b_sq = acc_b * acc_b
        norm_tile = tl.sum(tl.where(mask_tile, acc_b_sq, 0.0))
        norm_ptr = Norms_ptr + b * stride_norms_b
        tl.atomic_add(norm_ptr, norm_tile)

    offs_dw = offs_n[:, None] * stride_dw_out + offs_m[None, :] * stride_dw_in
    tl.store(DW_ptr + offs_dw, acc_global, mask=mask_n[:, None] & mask_m[None, :])


def run_triton_no_tma(
    x: torch.Tensor,
    grad_out: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Triton kernel without TMA/block pointers or Split-T.
    Uses naive strided loads; intended as non-TMA baseline.
    """
    B, T, Din = x.shape
    _, _, Dout = grad_out.shape
    if not x.is_contiguous():
        x = x.contiguous()
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    norms_buf = torch.zeros((B,), device=x.device, dtype=torch.float32)
    grad_weight = torch.empty((Dout, Din), device=x.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (triton.cdiv(Din, BLOCK_M) * triton.cdiv(Dout, BLOCK_N),)
    _fused_backward_kernel_no_tma[grid](
        x, grad_out, grad_weight, norms_buf,
        B, T, Din, Dout,
        x.stride(0), x.stride(1), x.stride(2),
        grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
        grad_weight.stride(0), grad_weight.stride(1),
        norms_buf.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return grad_weight, norms_buf


def run_triton_tma(
    x: torch.Tensor,
    grad_out: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused Triton kernel (uses TMA/block pointers; no Split-T)."""
    norms_buf = torch.zeros((x.shape[0],), device=x.device, dtype=torch.float32)
    grad_weight = tkernel.fused_backward_weight(
        x,
        grad_out,
        norms_buf,
        use_dsmem=False,  # Disable Split-T
    )
    return grad_weight, norms_buf


def run_triton_splitk(
    x: torch.Tensor,
    grad_out: torch.Tensor,
    split_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split-T Triton kernel."""
    norms_buf = torch.zeros((x.shape[0],), device=x.device, dtype=torch.float32)
    grad_weight = tkernel.fused_backward_weight_dsmem(
        x,
        grad_out,
        norms_buf,
        split_k=split_k,
    )
    return grad_weight, norms_buf


def maybe_check_triton() -> None:
    if not tkernel.TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available; install it to run this ablation.")


def build_variants(split_k: int) -> Dict[str, Dict]:
    return {
        "triton_no_tma": {
            "fn": run_triton_no_tma,
            "use_tma": False,
            "split_k": 0,
        },
        "triton_tma": {
            "fn": run_triton_tma,
            "use_tma": True,
            "split_k": 0,
        },
        "triton_splitk": {
            "fn": lambda x, g: run_triton_splitk(x, g, split_k=split_k),
            "use_tma": True,
            "split_k": split_k,
        },
    }


def plot_results(csv_path: Path, png_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plot.")
        return

    rows = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    labels = [r["variant"] for r in rows]
    means = [float(r["mean_ms"]) for r in rows]
    stds = [float(r["std_ms"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, means, yerr=stds, capsize=6, color=["#6aa84f", "#3d85c6", "#e69138"])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("FlashNorm Ablation (B=1, T=8196, Din=Dout=1024)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2.0, mean, f"{mean:.1f}", ha="center", va="bottom")
    plt.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="FlashNorm ablation: TMA + Split-T benefits.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "out")
    parser.add_argument("--device", type=str, default="cuda", help="CUDA device to benchmark on.")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--split-k", type=int, default=4, choices=[2, 4], help="Split factor for Split-T variant.")
    parser.add_argument("--shape", type=str, default="1,8196,1024,1024", help="B,T,Din,Dout (commas).")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this ablation.")

    maybe_check_triton()

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    B, T, Din, Dout = [int(x) for x in args.shape.split(",")]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "ablation_flashnorm.csv"
    png_path = args.output_dir / "ablation_flashnorm.png"

    x, grad_out = make_inputs(device=device, B=B, T=T, Din=Din, Dout=Dout, dtype=dtype)
    print(x.shape, grad_out.shape)
    variants = build_variants(split_k=args.split_k)

    results = []
    tokens = B * T
    reference = None

    with torch.no_grad():
        for name, cfg in variants.items():
            fn = cfg["fn"]

            def call_fn():
                return fn(x, grad_out)

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)

            mean_ms, std_ms = _time_fn(call_fn, args.iters, args.warmup)
            max_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            max_reserved = torch.cuda.max_memory_reserved(device) / (1024 * 1024)

            # Accuracy comparison vs reference (first variant)
            out_gw, out_norm = call_fn()
            if reference is None:
                reference = (out_gw.detach(), out_norm.detach())
                max_abs_err = 0.0
                rel_l2_err = 0.0
            else:
                ref_gw, ref_norm = reference
                gw_diff = (out_gw - ref_gw).float()
                norm_diff = (out_norm - ref_norm).float()
                gw_rel = torch.linalg.norm(gw_diff) / (torch.linalg.norm(ref_gw) + 1e-12)
                norm_rel = torch.linalg.norm(norm_diff) / (torch.linalg.norm(ref_norm) + 1e-12)
                rel_l2_err = float(torch.max(gw_rel, norm_rel))
                max_abs_err = float(torch.max(torch.abs(gw_diff)).item() if gw_diff.numel() > 0 else 0.0)
                max_abs_err = float(max(max_abs_err, torch.max(torch.abs(norm_diff)).item() if norm_diff.numel() > 0 else 0.0))

            tokens_per_s = tokens / (mean_ms / 1_000.0)
            results.append(
                BenchmarkResult(
                    variant=name,
                    mean_ms=mean_ms,
                    std_ms=std_ms,
                    tokens_per_s=tokens_per_s,
                    use_tma=cfg["use_tma"],
                    split_k=cfg["split_k"],
                    max_abs_err=max_abs_err,
                    rel_l2_err=rel_l2_err,
                    max_mem_mb=max_mem,
                    max_reserved_mb=max_reserved,
                )
            )
            print(
                f"{name:16s} mean={mean_ms:.3f} ms std={std_ms:.3f} ms "
                f"throughput={tokens_per_s:.1f} tok/s "
                f"peak_mem={max_mem:.1f}MB reserved={max_reserved:.1f}MB "
                f"max_abs_err={max_abs_err:.2e} rel_l2_err={rel_l2_err:.2e}"
            )

    # Write CSV
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "mean_ms",
                "std_ms",
                "tokens_per_s",
                "use_tma",
                "split_k",
                "max_abs_err",
                "rel_l2_err",
                "max_mem_mb",
                "max_reserved_mb",
            ],
        )
        writer.writeheader()
        for res in results:
            writer.writerow(
                {
                    "variant": res.variant,
                    "mean_ms": f"{res.mean_ms:.6f}",
                    "std_ms": f"{res.std_ms:.6f}",
                    "tokens_per_s": f"{res.tokens_per_s:.2f}",
                    "use_tma": res.use_tma,
                    "split_k": res.split_k,
                    "max_abs_err": f"{res.max_abs_err:.4e}",
                    "rel_l2_err": f"{res.rel_l2_err:.4e}",
                    "max_mem_mb": f"{res.max_mem_mb:.2f}",
                    "max_reserved_mb": f"{res.max_reserved_mb:.2f}",
                }
            )

    plot_results(csv_path, png_path)
    print(f"Saved CSV to {csv_path}")
    print(f"Saved plot to {png_path}")


if __name__ == "__main__":
    main()


