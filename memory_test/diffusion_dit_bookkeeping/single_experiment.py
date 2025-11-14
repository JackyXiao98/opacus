#!/usr/bin/env python3
"""
Run a single memory profiling experiment for DiT in isolation.
This script is designed to be called from a shell script with different arguments.
"""

import argparse
import json
import sys
import os
import gc
import torch
import torch.nn as nn

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))

from opacus.grad_sample import GradSampleModuleFastGradientClipping
from opacus.optimizers import DPOptimizerFastGradientClipping
from opacus.utils.fast_gradient_clipping_utils import DPLossFastGradientClipping
from memory_test.diffusion_dit_bookkeeping.diffusion_dit_model import DiTModelWithFlashAttention
from memory_test.test_algo.detailed_memory_profiler import (
    EnhancedMemoryProfiler,
    print_memory_breakdown
)


def aggressive_cleanup():
    """Aggressive memory cleanup"""
    for _ in range(3):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    gc.collect(generation=2)


def run_vanilla_experiment(config, device, num_iter=3, warmup_iter=2):
    """Run Vanilla (no DP-SGD) experiment"""
    print(f"\n{'='*80}")
    print("EXPERIMENT: Vanilla (No DP-SGD)")
    print(f"{'='*80}\n")
    
    aggressive_cleanup()
    
    # Create model
    model = DiTModelWithFlashAttention(
        img_size=config["image_size"],
        patch_size=config["patch_size"],
        in_channels=config["in_channels"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        num_classes=config["num_classes"],
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create profiler
    profiler = EnhancedMemoryProfiler(model, device)
    profiler.take_snapshot("0_model_loaded")
    
    # Warmup
    print(f"Running {warmup_iter} warmup iterations...")
    for i in range(warmup_iter):
        images = torch.randn(config["batch_size"], config["in_channels"], 
                           config["image_size"], config["image_size"], device=device)
        timesteps = torch.randint(0, 1000, (config["batch_size"],), device=device)
        labels = torch.randint(0, config["num_classes"], (config["batch_size"],), device=device)
        target_noise = torch.randn_like(images)
        
        outputs = model(images, timesteps, labels, target_noise=target_noise)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        del loss, outputs, images, timesteps, labels, target_noise
        if device == "cuda":
            torch.cuda.empty_cache()
    
    # Reset stats after warmup
    profiler.reset()
    profiler.take_snapshot("1_after_warmup")
    
    # Actual profiling iterations
    print(f"\nRunning {num_iter} profiling iterations...")
    total_time = 0
    
    for i in range(num_iter):
        images = torch.randn(config["batch_size"], config["in_channels"], 
                           config["image_size"], config["image_size"], device=device)
        timesteps = torch.randint(0, 1000, (config["batch_size"],), device=device)
        labels = torch.randint(0, config["num_classes"], (config["batch_size"],), device=device)
        target_noise = torch.randn_like(images)
        
        profiler.take_snapshot(f"2_iter{i}_before_forward")
        
        # Forward
        if device == "cuda":
            torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        
        outputs = model(images, timesteps, labels, target_noise=target_noise)
        loss = outputs["loss"]
        
        profiler.take_snapshot(f"3_iter{i}_after_forward")
        
        # Backward
        loss.backward()
        
        profiler.take_snapshot(f"4_iter{i}_after_backward")
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        end_time.record()
        torch.cuda.synchronize()
        iter_time = start_time.elapsed_time(end_time)
        total_time += iter_time
        
        profiler.take_snapshot(f"5_iter{i}_after_step")
        
        del loss, outputs, images, timesteps, labels, target_noise
        if device == "cuda":
            torch.cuda.empty_cache()
    
    avg_time = total_time / num_iter
    print(f"\n⏱️  Average iteration time: {avg_time:.3f} ms")
    
    # Get final breakdown
    breakdown = profiler.get_detailed_breakdown(optimizer)
    print_memory_breakdown(breakdown)
    
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0

    # Save results
    results = {
        "experiment": "vanilla",
        "config": config,
        "avg_time_ms": avg_time,
        "peak_memory_mb": peak_mem_mb,
        "peak_allocated_memory_mb": breakdown.get("peak_allocated_mb", 0),
        "breakdown": breakdown,
        "snapshots": [s.to_dict() for s in profiler.snapshots]
    }
    
    # Cleanup
    profiler.clear_hooks()
    del model, optimizer, profiler
    aggressive_cleanup()
    
    return results


def run_dpsgd_experiment(config, device, use_flash_clipping=False, enable_bookkeeping=False, num_iter=3, warmup_iter=2):
    """Run DP-SGD experiment (Ghost, Flash Clipping, or Bookkeeping)"""
    if enable_bookkeeping:
        exp_name = "Bookkeeping"
    else:
        exp_name = "Flash Clipping" if use_flash_clipping else "Ghost Clipping"

    if use_flash_clipping and enable_bookkeeping:
        exp_name = "Flash Clipping w/ Bookkeeping"

    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'='*80}\n")
    
    aggressive_cleanup()
    
    # Create model
    model = DiTModelWithFlashAttention(
        img_size=config["image_size"],
        patch_size=config["patch_size"],
        in_channels=config["in_channels"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        num_classes=config["num_classes"],
    ).to(device)
    
    # Create profiler BEFORE wrapping with GradSampleModule
    profiler = EnhancedMemoryProfiler(model, device)
    profiler.take_snapshot("0_model_loaded")
    
    # Wrap with DP-SGD
    model = GradSampleModuleFastGradientClipping(
        model,
        use_flash_clipping=use_flash_clipping,
        use_ghost_clipping=True,  # All fast gradient clipping methods use this
        enable_fastdp_bookkeeping=enable_bookkeeping,
        loss_reduction="mean",
    )
    
    profiler.model = model  # Update reference
    profiler.take_snapshot("1_wrapped_with_dp")
    
    # Register component-level hooks
    profiler.register_component_hooks()
    
    # Create optimizer
    base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer = DPOptimizerFastGradientClipping(
        optimizer=base_optimizer,
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        expected_batch_size=config["batch_size"],
        loss_reduction="mean",
    )

    # Create loss wrapper - we'll use MSE loss directly in the model
    # but still need DPLoss for proper gradient handling
    criterion = nn.MSELoss(reduction="mean")
    dp_loss = DPLossFastGradientClipping(
        model,
        optimizer,
        criterion,
        loss_reduction="mean",
    )
    
    profiler.take_snapshot("2_optimizer_created")
    
    # Warmup
    print(f"Running {warmup_iter} warmup iterations...")
    for i in range(warmup_iter):
        images = torch.randn(config["batch_size"], config["in_channels"], 
                           config["image_size"], config["image_size"], device=device)
        timesteps = torch.randint(0, 1000, (config["batch_size"],), device=device)
        labels = torch.randint(0, config["num_classes"], (config["batch_size"],), device=device)
        target_noise = torch.randn_like(images)
        
        # Forward pass - returns tensor when target_noise=None (DP-SGD mode)
        predicted_noise = model(images, timesteps, labels, target_noise=None)
        
        # Extract only the noise prediction (first in_channels)
        if predicted_noise.shape[1] > config["in_channels"]:
            predicted_noise = predicted_noise[:, :config["in_channels"], :, :]
        
        # Flatten tensors for DPLoss: (B, C, H, W) -> (B, C*H*W)
        batch_size = predicted_noise.shape[0]
        pred_flat = predicted_noise.view(batch_size, -1)
        target_flat = target_noise.view(batch_size, -1)
        
        # Compute loss using DPLoss with proper shape
        loss = dp_loss(pred_flat, target_flat, shape=(batch_size, pred_flat.shape[1]))
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        del loss, images, timesteps, labels, target_noise, predicted_noise, pred_flat, target_flat
        if device == "cuda":
            torch.cuda.empty_cache()
    
    # Reset stats after warmup
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    profiler.activation_memory = 0
    profiler.norm_sample_memory = 0
    profiler.take_snapshot("3_after_warmup")
    
    # Actual profiling iterations
    print(f"\nRunning {num_iter} profiling iterations...")
    total_time = 0
    
    for i in range(num_iter):
        images = torch.randn(config["batch_size"], config["in_channels"], 
                           config["image_size"], config["image_size"], device=device)
        timesteps = torch.randint(0, 1000, (config["batch_size"],), device=device)
        labels = torch.randint(0, config["num_classes"], (config["batch_size"],), device=device)
        target_noise = torch.randn_like(images)
        
        profiler.take_snapshot(f"4_iter{i}_before_forward")
        
        # Forward
        if device == "cuda":
            torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        
        # Forward pass - returns tensor when target_noise=None (DP-SGD mode)
        predicted_noise = model(images, timesteps, labels, target_noise=None)
        
        # Extract only the noise prediction
        if predicted_noise.shape[1] > config["in_channels"]:
            predicted_noise = predicted_noise[:, :config["in_channels"], :, :]
        
        # Flatten tensors for DPLoss: (B, C, H, W) -> (B, C*H*W)
        batch_size = predicted_noise.shape[0]
        pred_flat = predicted_noise.view(batch_size, -1)
        target_flat = target_noise.view(batch_size, -1)
        
        # Compute loss using DPLoss with proper shape
        loss = dp_loss(pred_flat, target_flat, shape=(batch_size, pred_flat.shape[1]))
        
        profiler.take_snapshot(f"5_iter{i}_after_forward")
        
        # Backward
        loss.backward()
        
        profiler.take_snapshot(f"6_iter{i}_after_backward")
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        end_time.record()
        torch.cuda.synchronize()
        iter_time = start_time.elapsed_time(end_time)
        total_time += iter_time
        
        profiler.take_snapshot(f"7_iter{i}_after_step")
        
        del loss, images, timesteps, labels, target_noise, predicted_noise, pred_flat, target_flat
        if device == "cuda":
            torch.cuda.empty_cache()
    
    avg_time_ms = total_time / num_iter
    print(f"\n⏱️  Average iteration time: {avg_time_ms:.3f} ms")
    
    # Get final breakdown
    breakdown = profiler.get_detailed_breakdown(optimizer)
    print_memory_breakdown(breakdown)
    
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0

    results = {
        "peak_memory_mb": peak_mem_mb,
        "peak_allocated_memory_mb": breakdown.get("peak_allocated_mb", 0),
        "avg_time_ms": avg_time_ms,
        "breakdown": breakdown,
        "snapshots": [s.to_dict() for s in profiler.snapshots]
    }
    
    # Cleanup
    profiler.clear_hooks()
    del model, optimizer, profiler
    aggressive_cleanup()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run single memory profiling experiment for DiT")
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["vanilla", "ghost", "flash_clip", "bookkeeping", "flash_clip_bookkeeping"],
                       help="Which experiment to run")
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSON file path")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-iter", type=int, default=3)
    parser.add_argument("--warmup-iter", type=int, default=2)
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "in_channels": args.in_channels,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "num_classes": args.num_classes,
        "batch_size": args.batch_size
    }
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'#'*80}")
    print(f"Configuration: {config}")
    print(f"Device: {device}")
    print(f"{'#'*80}\n")
    
    # Run experiment
    if args.experiment == "vanilla":
        results = run_vanilla_experiment(config, device, args.num_iter, args.warmup_iter)
    elif args.experiment == "ghost":
        results = run_dpsgd_experiment(config, device, use_flash_clipping=False, 
                                      num_iter=args.num_iter, warmup_iter=args.warmup_iter)
    elif args.experiment == "flash_clip":
        results = run_dpsgd_experiment(config, device, use_flash_clipping=True,
                                      num_iter=args.num_iter, warmup_iter=args.warmup_iter)
    elif args.experiment == "bookkeeping":
        results = run_dpsgd_experiment(config, device, use_flash_clipping=False, enable_bookkeeping=True,
                                      num_iter=args.num_iter, warmup_iter=args.warmup_iter)
    elif args.experiment == "flash_clip_bookkeeping":
        results = run_dpsgd_experiment(config, device, use_flash_clipping=True, enable_bookkeeping=True,
                                      num_iter=args.num_iter, warmup_iter=args.warmup_iter)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ Experiment {args.experiment} finished!")
    print(f"✅ Peak Memory: {results['peak_memory_mb']:.2f} MB")
    print(f"✅ Peak Allocated Memory: {results['peak_allocated_memory_mb']:.2f} MB")
    print(f"✅ Average time per iteration: {results['avg_time_ms']:.2f} ms")
    print(f"{'='*80}\n")
    
    print(f"✅ Results saved to: {args.output}")
    print(f"✅ Avg Time: {results['avg_time_ms']:.2f} ms\n")


if __name__ == "__main__":
    main()

