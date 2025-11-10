#!/usr/bin/env python3
"""
Run a single memory profiling experiment in isolation.
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
from memory_test.test_algo.memory_profile_with_flash_attention import (
    SimpleBigModelWithFlashAttention,
    DPMultiheadAttentionWithFlashAttention
)
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
    model = SimpleBigModelWithFlashAttention(
        vocab_size=config["vocab_size"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        seq_len=config["seq_len"]
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create profiler
    profiler = EnhancedMemoryProfiler(model, device)
    profiler.take_snapshot("0_model_loaded")
    
    # Warmup
    print(f"Running {warmup_iter} warmup iterations...")
    for i in range(warmup_iter):
        input_ids = torch.randint(0, config["vocab_size"], 
                                  (config["batch_size"], config["seq_len"]), device=device)
        labels = torch.randint(0, config["vocab_size"], 
                              (config["batch_size"], config["seq_len"]), device=device)
        
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        del loss, outputs, input_ids, labels
        if device == "cuda":
            torch.cuda.empty_cache()
    
    # Reset stats after warmup
    profiler.reset()
    profiler.take_snapshot("1_after_warmup")
    
    # Actual profiling iterations
    print(f"\nRunning {num_iter} profiling iterations...")
    total_time = 0
    
    for i in range(num_iter):
        input_ids = torch.randint(0, config["vocab_size"], 
                                  (config["batch_size"], config["seq_len"]), device=device)
        labels = torch.randint(0, config["vocab_size"], 
                              (config["batch_size"], config["seq_len"]), device=device)
        
        profiler.take_snapshot(f"2_iter{i}_before_forward")
        
        # Forward
        if device == "cuda":
            torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        
        outputs = model(input_ids, labels=labels)
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
        
        del loss, outputs, input_ids, labels
        if device == "cuda":
            torch.cuda.empty_cache()
    
    avg_time = total_time / num_iter
    print(f"\n⏱️  Average iteration time: {avg_time:.3f} ms")
    
    # Get final breakdown
    breakdown = profiler.get_detailed_breakdown(optimizer)
    print_memory_breakdown(breakdown)
    
    # Save results
    results = {
        "experiment": "vanilla",
        "config": config,
        "avg_time_ms": avg_time,
        "peak_memory_mb": breakdown.get("peak_allocated_mb", 0),
        "breakdown": breakdown,
        "snapshots": [s.to_dict() for s in profiler.snapshots]
    }
    
    # Cleanup
    profiler.clear_hooks()
    del model, optimizer, profiler
    aggressive_cleanup()
    
    return results


def run_dpsgd_experiment(config, device, use_triton=False, num_iter=3, warmup_iter=2):
    """Run DP-SGD experiment (Ghost or Flash Clipping)"""
    exp_name = "Flash Clipping" if use_triton else "Ghost Clipping"
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'='*80}\n")
    
    aggressive_cleanup()
    
    # Create model
    model = SimpleBigModelWithFlashAttention(
        vocab_size=config["vocab_size"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        seq_len=config["seq_len"]
    ).to(device)
    
    # Create profiler BEFORE wrapping with GradSampleModule
    profiler = EnhancedMemoryProfiler(model, device)
    profiler.take_snapshot("0_model_loaded")
    
    # Wrap with DP-SGD
    model = GradSampleModuleFastGradientClipping(
        model,
        use_triton=use_triton,
        use_ghost_clipping=True  # Both use ghost clipping framework
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
    )
    
    profiler.take_snapshot("2_optimizer_created")
    
    # Warmup
    print(f"Running {warmup_iter} warmup iterations...")
    for i in range(warmup_iter):
        input_ids = torch.randint(0, config["vocab_size"], 
                                  (config["batch_size"], config["seq_len"]), device=device)
        labels = torch.randint(0, config["vocab_size"], 
                              (config["batch_size"], config["seq_len"]), device=device)
        
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        del loss, outputs, input_ids, labels
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
        input_ids = torch.randint(0, config["vocab_size"], 
                                  (config["batch_size"], config["seq_len"]), device=device)
        labels = torch.randint(0, config["vocab_size"], 
                              (config["batch_size"], config["seq_len"]), device=device)
        
        profiler.take_snapshot(f"4_iter{i}_before_forward")
        
        # Forward
        if device == "cuda":
            torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        
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
        
        del loss, outputs, input_ids, labels
        if device == "cuda":
            torch.cuda.empty_cache()
    
    avg_time = total_time / num_iter
    print(f"\n⏱️  Average iteration time: {avg_time:.3f} ms")
    
    # Get final breakdown
    breakdown = profiler.get_detailed_breakdown(optimizer)
    print_memory_breakdown(breakdown)
    
    # Save results
    exp_key = "flash_clip" if use_triton else "ghost"
    results = {
        "experiment": exp_key,
        "config": config,
        "avg_time_ms": avg_time,
        "peak_memory_mb": breakdown.get("peak_allocated_mb", 0),
        "breakdown": breakdown,
        "snapshots": [s.to_dict() for s in profiler.snapshots]
    }
    
    # Cleanup
    profiler.clear_hooks()
    del model, optimizer, profiler
    aggressive_cleanup()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run single memory profiling experiment")
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["vanilla", "ghost", "flash_clip"],
                       help="Which experiment to run")
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSON file path")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=20)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-iter", type=int, default=3)
    parser.add_argument("--warmup-iter", type=int, default=2)
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        "vocab_size": args.vocab_size,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "seq_len": args.seq_len,
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
        results = run_dpsgd_experiment(config, device, use_triton=False, 
                                      num_iter=args.num_iter, warmup_iter=args.warmup_iter)
    elif args.experiment == "flash_clip":
        results = run_dpsgd_experiment(config, device, use_triton=True,
                                      num_iter=args.num_iter, warmup_iter=args.warmup_iter)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Experiment '{args.experiment}' completed!")
    print(f"✅ Results saved to: {args.output}")
    print(f"✅ Peak Memory: {results['peak_memory_mb']:.2f} MB")
    print(f"✅ Avg Time: {results['avg_time_ms']:.2f} ms\n")


if __name__ == "__main__":
    main()

