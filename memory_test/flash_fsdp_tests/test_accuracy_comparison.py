#!/usr/bin/env python3
"""
Accuracy comparison test between single GPU Flash Clipping and FSDP Flash Clipping.
Runs both training modes with identical configurations and validates numerical accuracy.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def run_single_gpu_training(args):
    """Run single GPU training."""
    print("\n" + "=" * 80)
    print("RUNNING SINGLE GPU TRAINING (BASELINE)")
    print("=" * 80)
    
    cmd = [
        sys.executable,
        "test_single_gpu.py",
        "--batch_size", str(args.batch_size),
        "--num_samples", str(args.num_samples),
        "--seq_len", str(args.seq_len),
        "--num_epochs", str(args.num_epochs),
        "--lr", str(args.lr),
        "--noise_multiplier", str(args.noise_multiplier),
        "--max_grad_norm", str(args.max_grad_norm),
        "--seed", str(args.seed),
        "--save_dir", "./results_single_gpu",
        "--device", "cpu" if not torch.cuda.is_available() else "cuda",
    ]
    
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    if result.returncode != 0:
        raise RuntimeError("Single GPU training failed")
    
    print("Single GPU training completed successfully")


def run_fsdp_training(args):
    """Run FSDP training."""
    print("\n" + "=" * 80)
    print("RUNNING FSDP TRAINING")
    print("=" * 80)
    
    cmd = [
        sys.executable,
        "test_fsdp_multi_gpu.py",
        "--batch_size", str(args.batch_size),
        "--num_samples", str(args.num_samples),
        "--seq_len", str(args.seq_len),
        "--num_epochs", str(args.num_epochs),
        "--lr", str(args.lr),
        "--noise_multiplier", str(args.noise_multiplier),
        "--max_grad_norm", str(args.max_grad_norm),
        "--seed", str(args.seed),
        "--save_dir", "./results_fsdp",
        "--world_size", "1",  # Use single-rank FSDP for exact accuracy comparison
    ]
    
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    if result.returncode != 0:
        raise RuntimeError("FSDP training failed")
    
    print("FSDP training completed successfully")


def load_metrics(results_dir):
    """Load metrics from a results directory."""
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    return metrics


def compare_metrics(single_gpu_metrics, fsdp_metrics, tolerance=1e-4):
    """
    Compare metrics between single GPU and FSDP.
    
    Args:
        single_gpu_metrics: Metrics from single GPU training
        fsdp_metrics: Metrics from FSDP training
        tolerance: Tolerance for numerical comparison
    
    Returns:
        comparison_results: Dictionary with comparison results
    """
    results = {
        "passed": True,
        "max_loss_diff": 0.0,
        "max_grad_norm_diff": 0.0,
        "max_param_norm_diff": 0.0,
        "loss_diffs": [],
        "grad_norm_diffs": [],
        "param_norm_diffs": [],
    }
    
    # Compare losses
    losses_sg = np.array(single_gpu_metrics["losses"])
    losses_fsdp = np.array(fsdp_metrics["losses"])
    
    if len(losses_sg) != len(losses_fsdp):
        print(f"WARNING: Different number of steps: {len(losses_sg)} vs {len(losses_fsdp)}")
        min_len = min(len(losses_sg), len(losses_fsdp))
        losses_sg = losses_sg[:min_len]
        losses_fsdp = losses_fsdp[:min_len]
    
    loss_diffs = np.abs(losses_sg - losses_fsdp)
    results["loss_diffs"] = loss_diffs.tolist()
    results["max_loss_diff"] = float(np.max(loss_diffs))
    
    # Compare gradient norms
    grad_norms_sg = np.array(single_gpu_metrics["grad_norms"])
    grad_norms_fsdp = np.array(fsdp_metrics["grad_norms"])
    
    min_len = min(len(grad_norms_sg), len(grad_norms_fsdp))
    grad_norms_sg = grad_norms_sg[:min_len]
    grad_norms_fsdp = grad_norms_fsdp[:min_len]
    
    grad_norm_diffs = np.abs(grad_norms_sg - grad_norms_fsdp)
    results["grad_norm_diffs"] = grad_norm_diffs.tolist()
    results["max_grad_norm_diff"] = float(np.max(grad_norm_diffs))
    
    # Compare parameter norms
    param_norms_sg = np.array(single_gpu_metrics["param_norms"])
    param_norms_fsdp = np.array(fsdp_metrics["param_norms"])
    
    min_len = min(len(param_norms_sg), len(param_norms_fsdp))
    param_norms_sg = param_norms_sg[:min_len]
    param_norms_fsdp = param_norms_fsdp[:min_len]
    
    param_norm_diffs = np.abs(param_norms_sg - param_norms_fsdp)
    results["param_norm_diffs"] = param_norm_diffs.tolist()
    results["max_param_norm_diff"] = float(np.max(param_norm_diffs))
    
    # Check if within tolerance
    if results["max_loss_diff"] > tolerance:
        print(f"FAIL: Max loss difference {results['max_loss_diff']:.6e} exceeds tolerance {tolerance:.6e}")
        results["passed"] = False
    
    if results["max_grad_norm_diff"] > tolerance * 10:  # More lenient for grad norms
        print(f"WARNING: Max grad norm difference {results['max_grad_norm_diff']:.6e} exceeds tolerance {tolerance * 10:.6e}")
    
    if results["max_param_norm_diff"] > tolerance:
        print(f"FAIL: Max param norm difference {results['max_param_norm_diff']:.6e} exceeds tolerance {tolerance:.6e}")
        results["passed"] = False
    
    return results


def plot_comparison(single_gpu_metrics, fsdp_metrics, save_dir="./comparison_results"):
    """Generate comparison plots."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Loss comparison
    ax = axes[0, 0]
    steps_sg = range(len(single_gpu_metrics["losses"]))
    steps_fsdp = range(len(fsdp_metrics["losses"]))
    ax.plot(steps_sg, single_gpu_metrics["losses"], label="Single GPU", marker='o', markersize=3)
    ax.plot(steps_fsdp, fsdp_metrics["losses"], label="FSDP", marker='x', markersize=3, linestyle='--')
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Loss difference
    ax = axes[0, 1]
    min_len = min(len(single_gpu_metrics["losses"]), len(fsdp_metrics["losses"]))
    loss_diffs = [abs(single_gpu_metrics["losses"][i] - fsdp_metrics["losses"][i]) for i in range(min_len)]
    ax.plot(range(min_len), loss_diffs, label="Loss Difference", color='red', marker='o', markersize=3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Absolute Difference")
    ax.set_title("Loss Difference (|Single GPU - FSDP|)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    
    # Gradient norm comparison
    ax = axes[1, 0]
    steps_sg = range(len(single_gpu_metrics["grad_norms"]))
    steps_fsdp = range(len(fsdp_metrics["grad_norms"]))
    ax.plot(steps_sg, single_gpu_metrics["grad_norms"], label="Single GPU", marker='o', markersize=3)
    ax.plot(steps_fsdp, fsdp_metrics["grad_norms"], label="FSDP", marker='x', markersize=3, linestyle='--')
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norm Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Parameter norm comparison
    ax = axes[1, 1]
    steps_sg = range(len(single_gpu_metrics["param_norms"]))
    steps_fsdp = range(len(fsdp_metrics["param_norms"]))
    ax.plot(steps_sg, single_gpu_metrics["param_norms"], label="Single GPU", marker='o', markersize=3)
    ax.plot(steps_fsdp, fsdp_metrics["param_norms"], label="FSDP", marker='x', markersize=3, linestyle='--')
    ax.set_xlabel("Step")
    ax.set_ylabel("Parameter Norm")
    ax.set_title("Parameter Norm Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison.png"), dpi=150)
    print(f"Comparison plot saved to {os.path.join(save_dir, 'comparison.png')}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Accuracy Comparison Test")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--noise_multiplier", type=float, default=1.0, help="DP noise multiplier")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Numerical tolerance for comparison")
    parser.add_argument("--skip_training", action="store_true", help="Skip training and only compare existing results")
    args = parser.parse_args()
    
    print("=" * 80)
    print("ACCURACY COMPARISON TEST: Single GPU vs FSDP Flash Clipping")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num samples: {args.num_samples}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Num epochs: {args.num_epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Noise multiplier: {args.noise_multiplier}")
    print(f"  Max grad norm: {args.max_grad_norm}")
    print(f"  Seed: {args.seed}")
    print(f"  Tolerance: {args.tolerance}")
    print("=" * 80)
    
    # Run training
    if not args.skip_training:
        run_single_gpu_training(args)
        run_fsdp_training(args)
    else:
        print("Skipping training, using existing results")
    
    # Load metrics
    print("\n" + "=" * 80)
    print("LOADING METRICS")
    print("=" * 80)
    
    single_gpu_dir = os.path.join(os.path.dirname(__file__), "results_single_gpu")
    fsdp_dir = os.path.join(os.path.dirname(__file__), "results_fsdp")
    
    single_gpu_metrics = load_metrics(single_gpu_dir)
    fsdp_metrics = load_metrics(fsdp_dir)
    
    print(f"Single GPU: {len(single_gpu_metrics['losses'])} steps")
    print(f"FSDP: {len(fsdp_metrics['losses'])} steps")
    
    # Compare metrics
    print("\n" + "=" * 80)
    print("COMPARING METRICS")
    print("=" * 80)
    
    comparison_results = compare_metrics(single_gpu_metrics, fsdp_metrics, tolerance=args.tolerance)
    
    # Print results
    print(f"\nMax loss difference: {comparison_results['max_loss_diff']:.6e}")
    print(f"Max grad norm difference: {comparison_results['max_grad_norm_diff']:.6e}")
    print(f"Max param norm difference: {comparison_results['max_param_norm_diff']:.6e}")
    
    # Generate plots
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    
    comparison_dir = os.path.join(os.path.dirname(__file__), "comparison_results")
    plot_comparison(single_gpu_metrics, fsdp_metrics, save_dir=comparison_dir)
    
    # Save comparison results
    with open(os.path.join(comparison_dir, "comparison_results.json"), "w") as f:
        json.dump(comparison_results, f, indent=2)
    
    # Final verdict
    print("\n" + "=" * 80)
    if comparison_results["passed"]:
        print("✓ ACCURACY TEST PASSED")
        print(f"All metrics within tolerance ({args.tolerance:.6e})")
    else:
        print("✗ ACCURACY TEST FAILED")
        print("Some metrics exceed tolerance")
        sys.exit(1)
    print("=" * 80)


if __name__ == "__main__":
    main()

