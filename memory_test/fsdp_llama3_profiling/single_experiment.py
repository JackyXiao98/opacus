#!/usr/bin/env python3
"""
Run a single FSDP Llama3 memory profiling experiment.
This script is designed to be called from a shell script with different arguments.
"""

import argparse
import gc
import json
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from opacus import PrivacyEngine
from opacus.utils.fsdp_utils import FSDP2Wrapper
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def aggressive_cleanup():
    """Aggressive memory cleanup"""
    for _ in range(3):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    gc.collect(generation=2)


def generate_synthetic_batch(batch_size, seq_length, vocab_size, device, num_labels=3):
    """Generate synthetic random data batch for sequence classification"""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    labels = torch.randint(0, num_labels, (batch_size,), device=device)  # One label per sequence
    attention_mask = torch.ones((batch_size, seq_length), device=device, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def setup(rank, world_size):
    """Setup distributed training"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """Cleanup distributed training"""
    dist.destroy_process_group()


def run_experiment_worker(
    rank,
    world_size,
    mode,
    model_name,
    token,
    seq_length,
    batch_size,
    num_iter,
    warmup_iter,
    vocab_size,
    learning_rate,
    sigma,
    max_grad_norm,
    results_dict,
):
    """Run experiment on a single GPU worker"""
    torch.cuda.set_device(rank)
    setup(rank, world_size)
    
    master_process = rank == 0
    torch.manual_seed(1337 + rank)
    
    if master_process:
        print(f"\n{'='*80}")
        print(f"EXPERIMENT: mode={mode}, seq_length={seq_length}, batch_size={batch_size}")
        print(f"{'='*80}\n")
    
    # Load model and tokenizer
    if master_process:
        print(f"Loading model: {model_name}")
    
    try:
        from huggingface_hub import login
        login(token)
    except Exception as e:
        if master_process:
            print(f"Warning: Could not login to HuggingFace: {e}")
    
    # Load tokenizer to get pad_token_id
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load pretrained model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,  # For classification task
    )
    
    # Set pad_token_id in model config
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Wrap with FSDP2
    mp_policy = dist.fsdp.MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, 
        reduce_dtype=torch.float32
    )
    model = FSDP2Wrapper(model, mp_policy=mp_policy)
    model.train()
    
    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Apply DP-SGD if needed
    if mode in ["ghost_fsdp", "flash_fsdp"]:
        if master_process:
            print(f"Applying DP-SGD with mode: {mode}")
        
        privacy_engine = PrivacyEngine()
        
        # Create dummy dataloader for make_private
        from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
        dummy_data = torch.randn(batch_size * world_size, seq_length)
        dummy_labels = torch.randint(0, 3, (batch_size * world_size,))
        dummy_dataset = TensorDataset(dummy_data, dummy_labels)
        dummy_dataloader = DataLoader(
            dummy_dataset,
            batch_size=batch_size,
            sampler=DistributedSampler(dummy_dataset, num_replicas=world_size, rank=rank),
        )
        
        model, optimizer, criterion, _ = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dummy_dataloader,
            noise_multiplier=sigma,
            max_grad_norm=max_grad_norm,
            grad_sample_mode=mode,
            criterion=criterion,
            poisson_sampling=False,
        )
        
        if master_process:
            print("DP-SGD setup complete")
    else:
        if master_process:
            print("Running in non-DP mode")
    
    device = torch.device(f"cuda:{rank}")
    
    # Warmup iterations
    if master_process:
        print(f"\nRunning {warmup_iter} warmup iterations...")
    
    for i in range(warmup_iter):
        batch = generate_synthetic_batch(batch_size, seq_length, vocab_size, device)
        
        optimizer.zero_grad()
        
        # Model computes loss internally when labels are provided
        if mode in ["ghost_fsdp", "flash_fsdp"]:
            # For DP modes, we need to use the criterion wrapper
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            loss = criterion(outputs.logits, batch["labels"])
        else:
            # For non-DP mode, model computes loss internally
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
        del batch, outputs, loss
        torch.cuda.empty_cache()
    
    # Reset memory stats after warmup
    torch.cuda.reset_peak_memory_stats(device)
    
    # Profiling iterations
    if master_process:
        print(f"\nRunning {num_iter} profiling iterations...")
    
    total_time = 0.0
    
    for i in range(num_iter):
        batch = generate_synthetic_batch(batch_size, seq_length, vocab_size, device)
        
        # Time the iteration
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        
        optimizer.zero_grad()
        
        # Model computes loss internally when labels are provided
        if mode in ["ghost_fsdp", "flash_fsdp"]:
            # For DP modes, we need to use the criterion wrapper
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            loss = criterion(outputs.logits, batch["labels"])
        else:
            # For non-DP mode, model computes loss internally
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
        end_event.record()
        torch.cuda.synchronize()
        
        iter_time = start_event.elapsed_time(end_event)
        total_time += iter_time
        
        if master_process:
            print(f"  Iteration {i+1}/{num_iter}: {iter_time:.2f} ms")
        
        del batch, outputs, loss
        torch.cuda.empty_cache()
    
    avg_time_ms = total_time / num_iter
    peak_memory_bytes = torch.cuda.max_memory_allocated(device)
    peak_memory_mb = peak_memory_bytes / (1024 ** 2)
    peak_memory_gb = peak_memory_bytes / (1024 ** 3)
    
    if master_process:
        print(f"\n⏱️  Average iteration time: {avg_time_ms:.2f} ms")
        print(f"💾 Peak memory usage: {peak_memory_gb:.2f} GB ({peak_memory_mb:.2f} MB)")
    
    # Store results in shared dict (only from rank 0)
    if master_process:
        results_dict["peak_memory_mb"] = peak_memory_mb
        results_dict["peak_memory_gb"] = peak_memory_gb
        results_dict["avg_time_ms"] = avg_time_ms
    
    cleanup()


def run_experiment(config):
    """Run the experiment with the given configuration"""
    world_size = torch.cuda.device_count()
    
    if world_size == 0:
        raise RuntimeError("No CUDA devices available")
    
    print(f"\n{'#'*80}")
    print(f"Configuration:")
    print(f"  Mode: {config['mode']}")
    print(f"  Model: {config['model_name']}")
    print(f"  Sequence Length: {config['seq_length']}")
    print(f"  Batch Size (per GPU): {config['batch_size']}")
    print(f"  Number of GPUs: {world_size}")
    print(f"  Total Batch Size: {config['batch_size'] * world_size}")
    print(f"  Profiling Iterations: {config['num_iter']}")
    print(f"  Warmup Iterations: {config['warmup_iter']}")
    print(f"{'#'*80}\n")
    
    # Use multiprocessing manager for sharing results
    manager = mp.Manager()
    results_dict = manager.dict()
    
    # Spawn processes
    mp.spawn(
        run_experiment_worker,
        args=(
            world_size,
            config["mode"],
            config["model_name"],
            config["token"],
            config["seq_length"],
            config["batch_size"],
            config["num_iter"],
            config["warmup_iter"],
            config["vocab_size"],
            config["learning_rate"],
            config["sigma"],
            config["max_grad_norm"],
            results_dict,
        ),
        nprocs=world_size,
        join=True,
    )
    
    # Extract results
    results = {
        "mode": config["mode"],
        "seq_length": config["seq_length"],
        "batch_size": config["batch_size"],
        "total_batch_size": config["batch_size"] * world_size,
        "num_gpus": world_size,
        "peak_memory_mb": results_dict.get("peak_memory_mb", 0),
        "peak_memory_gb": results_dict.get("peak_memory_gb", 0),
        "avg_time_ms": results_dict.get("avg_time_ms", 0),
        "config": {
            "model_name": config["model_name"],
            "vocab_size": config["vocab_size"],
            "learning_rate": config["learning_rate"],
            "sigma": config["sigma"],
            "max_grad_norm": config["max_grad_norm"],
        }
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run single FSDP Llama3 profiling experiment")
    parser.add_argument("--mode", type=str, required=True,
                       choices=["no_dp", "ghost_fsdp", "flash_fsdp"],
                       help="Training mode")
    parser.add_argument("--seq-length", type=int, required=True,
                       help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size per GPU")
    parser.add_argument("--num-iter", type=int, default=3,
                       help="Number of profiling iterations")
    parser.add_argument("--warmup-iter", type=int, default=1,
                       help="Number of warmup iterations")
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSON file path")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Model name")
    parser.add_argument("--token", type=str, required=True,
                       help="HuggingFace token")
    parser.add_argument("--vocab-size", type=int, default=128256,
                       help="Vocabulary size for synthetic data")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--sigma", type=float, default=1.0,
                       help="Noise multiplier for DP")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                       help="Max gradient norm for DP")
    
    args = parser.parse_args()
    
    config = {
        "mode": args.mode,
        "seq_length": args.seq_length,
        "batch_size": args.batch_size,
        "num_iter": args.num_iter,
        "warmup_iter": args.warmup_iter,
        "model_name": args.model_name,
        "token": args.token,
        "vocab_size": args.vocab_size,
        "learning_rate": args.learning_rate,
        "sigma": args.sigma,
        "max_grad_norm": args.max_grad_norm,
    }
    
    # Run experiment
    aggressive_cleanup()
    results = run_experiment(config)
    aggressive_cleanup()
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ Experiment completed!")
    print(f"✅ Mode: {results['mode']}")
    print(f"✅ Seq Length: {results['seq_length']}")
    print(f"✅ Peak Memory: {results['peak_memory_gb']:.2f} GB")
    print(f"✅ Avg Time: {results['avg_time_ms']:.2f} ms")
    print(f"✅ Results saved to: {args.output}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

