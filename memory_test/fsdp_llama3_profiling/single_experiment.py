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
from opacus.utils.fast_gradient_clipping_utils import DPTensorFastGradientClipping
from opacus.utils.fsdp_utils import FSDP2Wrapper
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def compute_linear_layer_norms(model, prefix=""):
    """Compute L2 norms of all linear layer weights and gradients"""
    norms = {}
    for name, module in model.named_modules():
        # Handle both regular Linear and FusedFlashLinear
        if hasattr(module, 'weight') and 'linear' in name.lower() or isinstance(module, torch.nn.Linear):
            full_name = f"{prefix}{name}" if prefix else name
            if module.weight is not None:
                weight = module.weight
                # Handle DTensor for FSDP
                if hasattr(weight, 'full_tensor'):
                    weight = weight.full_tensor()
                norms[f"{full_name}_weight_norm"] = weight.detach().float().norm().item()
                
                if module.weight.grad is not None:
                    grad = module.weight.grad
                    if hasattr(grad, 'full_tensor'):
                        grad = grad.full_tensor()
                    norms[f"{full_name}_grad_norm"] = grad.detach().float().norm().item()
    return norms


def compute_total_grad_norm(model):
    """Compute total gradient norm across all parameters"""
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad = param.grad
            if hasattr(grad, 'full_tensor'):
                grad = grad.full_tensor()
            total_norm += grad.detach().float().norm().item() ** 2
    return total_norm ** 0.5


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
    accuracy_test=False,
    random_seed=1337,
):
    """Run experiment on a single GPU worker"""
    torch.cuda.set_device(rank)
    
    # Determine if this is an FSDP mode
    is_fsdp_mode = mode in ["ghost_fsdp", "flash_fsdp", "flash_fsdp_bk", "ghost_fsdp_bk", "flash_fsdp_fuse", "flash_fsdp_fuse_bk", "no_dp"]
    
    # Only setup distributed for FSDP modes
    if is_fsdp_mode:
        setup(rank, world_size)
    
    master_process = rank == 0
    torch.manual_seed(random_seed + rank)
    
    if master_process:
        print(f"\n{'='*80}")
        print(f"EXPERIMENT: mode={mode}, seq_length={seq_length}, batch_size={batch_size}")
        if accuracy_test:
            print(f"ACCURACY TEST MODE: sigma={sigma}, max_grad_norm={max_grad_norm}")
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
    
    # Determine if this is a fuse mode (needs special handling)
    is_fuse_mode = mode in ["flash_fsdp_fuse", "flash_fsdp_fuse_bk"]
    
    # For fuse modes: Replace Linear with FusedFlashLinear BEFORE FSDP wrapping
    # This is required because FusedFlashLinear cannot be created after FSDP
    # wrapping (DTensor weights cannot be copied to regular Tensors)
    if is_fuse_mode:
        if master_process:
            print("Replacing Linear layers with FusedFlashLinear (pre-FSDP)")
        from opacus.grad_sample.fused_flash_linear import replace_linear_with_fused
        model = replace_linear_with_fused(model, algorithm="input_length", tile_size=256)
    
    # Wrap with FSDP2 only for FSDP modes
    if is_fsdp_mode:
        if master_process:
            print("Wrapping model with FSDP2")
        mp_policy = dist.fsdp.MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, 
            reduce_dtype=torch.float32
        )
        model = FSDP2Wrapper(model, mp_policy=mp_policy)
    else:
        if master_process:
            print("Using single-GPU mode (no FSDP)")
        # Move model to device for single-GPU mode
        model = model.to(f"cuda:{rank}")
    
    model.train()
    
    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Apply DP-SGD if needed (exclude no_dp and no_dp_single)
    is_dp_mode = mode not in ["no_dp", "no_dp_single"]
    
    if is_dp_mode:
        if master_process:
            print(f"Applying DP-SGD with mode: {mode}")
        
        privacy_engine = PrivacyEngine()
        
        # Create dummy dataloader for make_private
        from torch.utils.data import TensorDataset, DataLoader
        
        if is_fsdp_mode:
            # FSDP mode: use DistributedSampler
            from torch.utils.data import DistributedSampler
            dummy_data = torch.randn(batch_size * world_size, seq_length)
            dummy_labels = torch.randint(0, 3, (batch_size * world_size,))
            dummy_dataset = TensorDataset(dummy_data, dummy_labels)
            dummy_dataloader = DataLoader(
                dummy_dataset,
                batch_size=batch_size,
                sampler=DistributedSampler(dummy_dataset, num_replicas=world_size, rank=rank),
            )
        else:
            # Single-GPU mode: regular dataloader
            dummy_data = torch.randn(batch_size, seq_length)
            dummy_labels = torch.randint(0, 3, (batch_size,))
            dummy_dataset = TensorDataset(dummy_data, dummy_labels)
            dummy_dataloader = DataLoader(
                dummy_dataset,
                batch_size=batch_size,
                shuffle=False,
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
        if is_dp_mode:
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
        
        # Synchronize in FSDP mode to avoid deadlocks
        if is_fsdp_mode:
            torch.cuda.synchronize()
            if dist.is_initialized():
                dist.barrier()
        
        del batch, outputs, loss
        torch.cuda.empty_cache()
    
    # Synchronize all ranks before profiling
    if is_fsdp_mode and dist.is_initialized():
        dist.barrier()
    
    # Reset memory stats after warmup
    torch.cuda.reset_peak_memory_stats(device)
    
    # Profiling iterations
    if master_process:
        print(f"\nRunning {num_iter} profiling iterations...")
    
    total_time = 0.0
    loss_values = []
    grad_norms = []
    layer_norms_history = []
    
    for i in range(num_iter):
        batch = generate_synthetic_batch(batch_size, seq_length, vocab_size, device)
        
        # Time the iteration
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        
        optimizer.zero_grad()
        
        # Model computes loss internally when labels are provided
        if is_dp_mode:
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
        
        # Record loss value (average across ranks for FSDP non-DP mode)
        if isinstance(loss, DPTensorFastGradientClipping):
            # DPTensor has its own .item() method that handles reduction
            loss_val = loss.item()
        elif is_fsdp_mode and not is_dp_mode and dist.is_initialized():
            # Average loss across all ranks for regular tensor
            loss_tensor = loss.detach().float()
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            loss_val = loss_tensor.item()
        else:
            # Regular tensor extraction
            loss_val = loss.detach().float().item()
        loss_values.append(loss_val)
        
        loss.backward()
        
        # For accuracy test, compute gradient norms before optimizer step
        if accuracy_test:
            # Synchronize before computing norms in FSDP mode
            if is_fsdp_mode and dist.is_initialized():
                dist.barrier()
            
            if master_process:
                # Get the underlying model for norm computation
                if hasattr(model, '_module'):
                    base_model = model._module
                else:
                    base_model = model
                total_grad_norm = compute_total_grad_norm(base_model)
                grad_norms.append(total_grad_norm)
                
                # Compute layer norms (only first and last iteration to save time)
                if i == 0 or i == num_iter - 1:
                    layer_norms = compute_linear_layer_norms(base_model)
                    layer_norms_history.append({"iter": i, "norms": layer_norms})
        
        optimizer.step()
        
        end_event.record()
        torch.cuda.synchronize()
        
        # Synchronize in FSDP mode to avoid deadlocks
        if is_fsdp_mode and dist.is_initialized():
            dist.barrier()
        
        iter_time = start_event.elapsed_time(end_event)
        total_time += iter_time
        
        if master_process:
            if accuracy_test:
                grad_norm_str = f", grad_norm={grad_norms[-1]:.6f}" if grad_norms else ""
                print(f"  Iteration {i+1}/{num_iter}: {iter_time:.2f} ms, loss={loss_val:.6f}{grad_norm_str}")
            else:
                print(f"  Iteration {i+1}/{num_iter}: {iter_time:.2f} ms")
        
        del batch, outputs, loss
        torch.cuda.empty_cache()
    
    avg_time_ms = total_time / num_iter
    peak_memory_bytes = torch.cuda.max_memory_allocated(device)
    peak_memory_mb = peak_memory_bytes / (1024 ** 2)
    peak_memory_gb = peak_memory_bytes / (1024 ** 3)
    
    # Synchronize before printing final results
    if is_fsdp_mode and dist.is_initialized():
        dist.barrier()
    
    if master_process:
        print(f"\n⏱️  Average iteration time: {avg_time_ms:.2f} ms")
        print(f"💾 Peak memory usage: {peak_memory_gb:.2f} GB ({peak_memory_mb:.2f} MB)")
        
        if accuracy_test:
            print(f"\n📊 ACCURACY TEST RESULTS:")
            print(f"   Loss values: {loss_values}")
            print(f"   Final loss: {loss_values[-1]:.6f}")
            if grad_norms:
                print(f"   Gradient norms: {grad_norms}")
                print(f"   Avg gradient norm: {sum(grad_norms)/len(grad_norms):.6f}")
    
    # Synchronize before storing results
    if is_fsdp_mode and dist.is_initialized():
        dist.barrier()
    
    # Store results in shared dict (only from rank 0)
    if master_process:
        results_dict["peak_memory_mb"] = peak_memory_mb
        results_dict["peak_memory_gb"] = peak_memory_gb
        results_dict["avg_time_ms"] = avg_time_ms
        results_dict["loss_values"] = loss_values
        results_dict["final_loss"] = loss_values[-1] if loss_values else 0
        if accuracy_test:
            results_dict["grad_norms"] = grad_norms
            results_dict["avg_grad_norm"] = sum(grad_norms)/len(grad_norms) if grad_norms else 0
            results_dict["layer_norms_history"] = layer_norms_history
    
    # Final synchronization before cleanup
    if is_fsdp_mode and dist.is_initialized():
        dist.barrier()
    
    # Only cleanup distributed for FSDP modes
    if is_fsdp_mode:
        cleanup()


def run_experiment(config):
    """Run the experiment with the given configuration"""
    # Determine if this is an FSDP mode (multi-GPU)
    is_fsdp_mode = config["mode"] in ["ghost_fsdp", "flash_fsdp", "flash_fsdp_bk", "ghost_fsdp_bk", "flash_fsdp_fuse", "flash_fsdp_fuse_bk", "no_dp"]
    
    if is_fsdp_mode:
        # FSDP mode: use all available GPUs
        world_size = torch.cuda.device_count()
        if world_size == 0:
            raise RuntimeError("No CUDA devices available")
    else:
        # Single-GPU mode: use only 1 GPU
        world_size = 1
        if torch.cuda.device_count() == 0:
            raise RuntimeError("No CUDA devices available")
    
    accuracy_test = config.get("accuracy_test", False)
    random_seed = config.get("random_seed", 1337)
    
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
    if accuracy_test:
        print(f"  ACCURACY TEST: sigma={config['sigma']}, max_grad_norm={config['max_grad_norm']}")
        print(f"  Random Seed: {random_seed}")
    print(f"{'#'*80}\n")
    
    # Use multiprocessing manager for sharing results
    manager = mp.Manager()
    results_dict = manager.dict()
    
    if is_fsdp_mode:
        # FSDP mode: spawn multiple processes
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
                accuracy_test,
                random_seed,
            ),
            nprocs=world_size,
            join=True,
        )
    else:
        # Single-GPU mode: run directly without multiprocessing
        run_experiment_worker(
            rank=0,
            world_size=1,
            mode=config["mode"],
            model_name=config["model_name"],
            token=config["token"],
            seq_length=config["seq_length"],
            batch_size=config["batch_size"],
            num_iter=config["num_iter"],
            warmup_iter=config["warmup_iter"],
            vocab_size=config["vocab_size"],
            learning_rate=config["learning_rate"],
            sigma=config["sigma"],
            max_grad_norm=config["max_grad_norm"],
            results_dict=results_dict,
            accuracy_test=accuracy_test,
            random_seed=random_seed,
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
        "loss_values": results_dict.get("loss_values", []),
        "final_loss": results_dict.get("final_loss", 0),
        "config": {
            "model_name": config["model_name"],
            "vocab_size": config["vocab_size"],
            "learning_rate": config["learning_rate"],
            "sigma": config["sigma"],
            "max_grad_norm": config["max_grad_norm"],
            "random_seed": config.get("random_seed", 1337),
        }
    }
    
    # Add accuracy test data if available
    if accuracy_test:
        results["grad_norms"] = results_dict.get("grad_norms", [])
        results["avg_grad_norm"] = results_dict.get("avg_grad_norm", 0)
        results["layer_norms_history"] = results_dict.get("layer_norms_history", [])
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run single FSDP/Single-GPU Llama3 profiling experiment")
    parser.add_argument("--mode", type=str, required=True,
                       choices=["no_dp", "no_dp_single", "ghost_fsdp", "flash_fsdp", "flash_fsdp_bk", "ghost_fsdp_bk", "flash_fsdp_fuse", "flash_fsdp_fuse_bk", "flash", "flash_bk", "ghost", "ghost_bk"],
                       help="Training mode: no_dp (multi-GPU no DP), no_dp_single (single-GPU no DP), *_fsdp (multi-GPU with DP), others (single-GPU with DP)")
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
    parser.add_argument("--accuracy-test", action="store_true",
                       help="Run accuracy/consistency test (logs loss and gradient norms)")
    parser.add_argument("--random-seed", type=int, default=1337,
                       help="Random seed for reproducibility")
    
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
        "accuracy_test": args.accuracy_test,
        "random_seed": args.random_seed,
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

