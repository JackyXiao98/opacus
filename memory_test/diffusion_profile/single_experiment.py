#!/usr/bin/env python3
"""
Run a single memory profiling experiment for HuggingFace DiT in isolation.
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

from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from memory_test.diffusion_profile.dit_huggingface_wrapper import DiTHuggingFaceWrapper
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


class DiTSyntheticDataset(torch.utils.data.Dataset):
    """Synthetic dataset for DiT profiling"""
    def __init__(self, config, device, num_samples=100):
        self.config = config
        self.device = device
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate synthetic diffusion data
        images = torch.randn(self.config["in_channels"], 
                           self.config["image_size"], 
                           self.config["image_size"])
        timesteps = torch.randint(0, 1000, (1,)).item()
        labels = torch.randint(0, self.config["num_classes"], (1,)).item()
        target_noise = torch.randn_like(images)
        
        return images, timesteps, labels, target_noise


def create_dit_criterion():
    """
    Create a criterion for DiT that computes per-sample MSE loss.
    Needed for PrivacyEngine with ghost/flash modes.
    """
    def dit_criterion(predicted, target):
        """
        Custom criterion for DiT that flattens outputs before computing loss.
        Args:
            predicted: (B, C, H, W) - predicted noise
            target: (B, C, H, W) - target noise
        Returns:
            loss_per_sample: (B,) - per-sample MSE loss
        """
        batch_size = predicted.shape[0]
        # Flatten spatial dimensions: (B, C, H, W) -> (B, C*H*W)
        pred_flat = predicted.reshape(batch_size, -1)
        target_flat = target.reshape(batch_size, -1)
        # Compute per-sample MSE and reduce over features
        loss_per_element = nn.functional.mse_loss(pred_flat, target_flat, reduction='none')
        return loss_per_element.mean(dim=1)  # (B,)
    
    # Set reduction attribute (required by PrivacyEngine)
    dit_criterion.reduction = "mean"
    return dit_criterion


def run_vanilla_experiment(config, device, num_iter=3, warmup_iter=2):
    """Run Vanilla (no DP-SGD) experiment"""
    print(f"\n{'='*80}")
    print("EXPERIMENT: Vanilla (No DP-SGD)")
    print(f"{'='*80}\n")
    
    aggressive_cleanup()
    
    # Create model using HuggingFace wrapper
    model = DiTHuggingFaceWrapper(
        model_name="microsoft/dit-large",
        img_size=config["image_size"],
        patch_size=config["patch_size"],
        in_channels=config["in_channels"],
        num_classes=config["num_classes"],
        pretrained=False,  # Use config only for faster loading
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
    """Run DP-SGD experiment using PrivacyEngine with ghost/flash/bookkeeping modes"""
    # Determine experiment name and grad_sample_mode
    if use_flash_clipping and enable_bookkeeping:
        exp_name = "Flash Clipping w/ Bookkeeping"
        grad_sample_mode = "flash_bk"
    elif use_flash_clipping:
        exp_name = "Flash Clipping"
        grad_sample_mode = "flash"
    elif enable_bookkeeping:
        exp_name = "Bookkeeping"
        grad_sample_mode = "ghost_bk"
    else:
        exp_name = "Ghost Clipping"
        grad_sample_mode = "ghost"

    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {exp_name} (grad_sample_mode={grad_sample_mode})")
    print(f"{'='*80}\n")
    
    aggressive_cleanup()
    
    # Create model using HuggingFace wrapper
    model = DiTHuggingFaceWrapper(
        model_name="microsoft/dit-large",
        img_size=config["image_size"],
        patch_size=config["patch_size"],
        in_channels=config["in_channels"],
        num_classes=config["num_classes"],
        pretrained=False,  # Use config only for faster loading
    ).to(device)
    
    # Create profiler BEFORE wrapping with PrivacyEngine
    profiler = EnhancedMemoryProfiler(model, device)
    profiler.take_snapshot("0_model_loaded")
    
    # Create optimizer
    base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create DataLoader with synthetic data (required by PrivacyEngine)
    dataset = DiTSyntheticDataset(config, device, num_samples=100)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    
    # Create criterion for per-sample loss computation
    criterion = create_dit_criterion()
    
    # Create PrivacyEngine and make model private
    privacy_engine = PrivacyEngine()
    
    print(f"Making model private with grad_sample_mode={grad_sample_mode}...")
    # For ghost/flash modes, make_private returns (model, optimizer, criterion, dataloader)
    # For other modes, it returns (model, optimizer, dataloader)
    result = privacy_engine.make_private(
        module=model,
        optimizer=base_optimizer,
        data_loader=dataloader,
        criterion=criterion,
        noise_multiplier=0.01,  # No noise for memory profiling
        max_grad_norm=1.0,
        grad_sample_mode=grad_sample_mode,
        poisson_sampling=False,  # Use regular batch sampling
    )
    
    # Unpack based on return length
    if len(result) == 4:
        # ghost/flash modes return (model, optimizer, criterion, dataloader)
        model, optimizer, criterion, private_dataloader = result
    else:
        # other modes return (model, optimizer, dataloader)
        model, optimizer, private_dataloader = result
    
    profiler.model = model  # Update reference
    profiler.take_snapshot("1_wrapped_with_dp")
    
    # Register component-level hooks
    profiler.register_component_hooks()
    profiler.take_snapshot("2_optimizer_created")
    
    # Warmup iterations
    print(f"Running {warmup_iter} warmup iterations...")
    warmup_count = 0
    for batch_data in private_dataloader:
        if warmup_count >= warmup_iter:
            break
        
        images, timesteps, labels, target_noise = batch_data
        images = images.to(device)
        timesteps = timesteps.to(device)
        labels = labels.to(device)
        target_noise = target_noise.to(device)
        
        # Forward pass - returns tensor directly in DP-SGD mode
        predicted_noise = model(images, timesteps, labels, target_noise=None)
        
        # Extract only the noise prediction (first in_channels) if needed
        if predicted_noise.shape[1] > config["in_channels"]:
            predicted_noise = predicted_noise[:, :config["in_channels"], :, :]
        
        # Compute loss using criterion
        # For ghost/flash modes, criterion returns DPTensorFastGradientClipping
        # which should be used directly without .mean()
        loss = criterion(predicted_noise, target_noise)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        warmup_count += 1
        
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
    iter_count = 0
    
    for batch_data in private_dataloader:
        if iter_count >= num_iter:
            break
        
        images, timesteps, labels, target_noise = batch_data
        images = images.to(device)
        timesteps = timesteps.to(device)
        labels = labels.to(device)
        target_noise = target_noise.to(device)
        
        profiler.take_snapshot(f"4_iter{iter_count}_before_forward")
        
        # Forward
        if device == "cuda":
            torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        
        # Forward pass - returns tensor directly in DP-SGD mode
        predicted_noise = model(images, timesteps, labels, target_noise=None)
        
        # Extract only the noise prediction if needed
        if predicted_noise.shape[1] > config["in_channels"]:
            predicted_noise = predicted_noise[:, :config["in_channels"], :, :]
        
        # Compute loss using criterion
        # For ghost/flash modes, criterion returns DPTensorFastGradientClipping
        # which should be used directly without .mean()
        loss = criterion(predicted_noise, target_noise)
        
        profiler.take_snapshot(f"5_iter{iter_count}_after_forward")
        
        # Backward
        loss.backward()
        
        profiler.take_snapshot(f"6_iter{iter_count}_after_backward")
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        end_time.record()
        torch.cuda.synchronize()
        iter_time = start_time.elapsed_time(end_time)
        total_time += iter_time
        
        profiler.take_snapshot(f"7_iter{iter_count}_after_step")
        
        iter_count += 1
        
        if device == "cuda":
            torch.cuda.empty_cache()
    
    avg_time_ms = total_time / num_iter if num_iter > 0 else 0
    print(f"\n⏱️  Average iteration time: {avg_time_ms:.3f} ms")
    
    # Get final breakdown
    breakdown = profiler.get_detailed_breakdown(optimizer)
    print_memory_breakdown(breakdown)
    
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0

    results = {
        "experiment": exp_name.lower().replace(" ", "_").replace("w/", ""),
        "config": config,
        "peak_memory_mb": peak_mem_mb,
        "peak_allocated_memory_mb": breakdown.get("peak_allocated_mb", 0),
        "avg_time_ms": avg_time_ms,
        "breakdown": breakdown,
        "snapshots": [s.to_dict() for s in profiler.snapshots]
    }
    
    # Cleanup
    profiler.clear_hooks()
    del model, optimizer, profiler, privacy_engine
    aggressive_cleanup()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run single memory profiling experiment for HuggingFace DiT")
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["vanilla", "ghost", "flash_clip", "bookkeeping", "flash_clip_bookkeeping"],
                       help="Which experiment to run")
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSON file path")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-iter", type=int, default=1)
    parser.add_argument("--warmup-iter", type=int, default=1)
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "in_channels": args.in_channels,
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

