#!/usr/bin/env python3
"""
Run a single FSDP memory profiling experiment.
This script is designed to be called from a shell script with different arguments.
Supports both LLM (HuggingFace) and DiT models.
"""

import argparse
import gc
import json
import os
import sys

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from opacus import PrivacyEngine
from opacus.utils.fast_gradient_clipping_utils import DPTensorFastGradientClipping
from opacus.utils.fsdp_utils import FSDP2Wrapper

# Add parent directories to path for DiT model imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))


def aggressive_cleanup():
    """Aggressive memory cleanup"""
    for _ in range(3):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    gc.collect(generation=2)


def create_llm_model(config, device, master_process=True):
    """Create LLM model from HuggingFace"""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    model_name = config["model_name"]
    token = config.get("token")
    
    if master_process:
        print(f"Loading LLM model: {model_name}")
    
    try:
        from huggingface_hub import login
        if token:
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
    
    return model


def create_dit_model(config, device, master_process=True):
    """Create DiT model"""
    from memory_test.diffusion_dit_bookkeeping.diffusion_dit_model import DiTModelWithFlashAttention
    
    if master_process:
        print(f"Creating DiT model: hidden_dim={config['hidden_dim']}, num_layers={config['num_layers']}")
    
    model = DiTModelWithFlashAttention(
        img_size=config["image_size"],
        patch_size=config["patch_size"],
        in_channels=config["in_channels"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        num_classes=config["num_classes"],
    )
    
    return model


def create_model(model_type, config, device, master_process=True):
    """Factory function to create model based on type"""
    if model_type == "llm":
        return create_llm_model(config, device, master_process)
    elif model_type == "dit":
        return create_dit_model(config, device, master_process)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def generate_llm_batch(config, device):
    """Generate synthetic batch for LLM"""
    batch_size = config["batch_size"]
    seq_length = config["seq_length"]
    vocab_size = config["vocab_size"]
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    labels = torch.randint(0, 3, (batch_size,), device=device)  # One label per sequence
    attention_mask = torch.ones((batch_size, seq_length), device=device, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def generate_dit_batch(config, device):
    """Generate synthetic batch for DiT"""
    batch_size = config["batch_size"]
    in_channels = config["in_channels"]
    image_size = config["image_size"]
    num_classes = config["num_classes"]
    
    images = torch.randn(batch_size, in_channels, image_size, image_size, device=device)
    timesteps = torch.randint(0, 1000, (batch_size,), device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    target_noise = torch.randn_like(images)
    
    return {"images": images, "timesteps": timesteps, "labels": labels, "target_noise": target_noise}


def generate_batch(model_type, config, device):
    """Factory function to generate batch based on model type"""
    if model_type == "llm":
        return generate_llm_batch(config, device)
    elif model_type == "dit":
        return generate_dit_batch(config, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


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
    model_type,
    mode,
    config,
    num_iter,
    warmup_iter,
    learning_rate,
    sigma,
    max_grad_norm,
    results_dict,
):
    """Run experiment on a single GPU worker"""
    torch.cuda.set_device(rank)
    
    # Determine if this is an FSDP mode
    is_fsdp_mode = mode in ["ghost_fsdp", "flash_fsdp", "flash_fsdp_bk", "ghost_fsdp_bk", "flash_fsdp_fuse", "flash_fsdp_fuse_bk", "no_dp"]
    
    # Only setup distributed for FSDP modes
    if is_fsdp_mode:
        setup(rank, world_size)
    
    master_process = rank == 0
    torch.manual_seed(1337 + rank)
    
    seq_length = config.get("seq_length", config.get("image_size", 0) // config.get("patch_size", 1) ** 2)
    batch_size = config["batch_size"]
    
    if master_process:
        print(f"\n{'='*80}")
        print(f"EXPERIMENT: model_type={model_type}, mode={mode}, seq_length={seq_length}, batch_size={batch_size}")
        print(f"{'='*80}\n")
    
    device = torch.device(f"cuda:{rank}")
    
    # Create model
    model = create_model(model_type, config, device, master_process)
    
    # Determine if this is a fuse mode (needs special handling)
    is_fuse_mode = mode in ["flash_fsdp_fuse", "flash_fsdp_fuse_bk", "flash_fuse", "flash_fuse_bk"]
    
    # For fuse modes: Replace Linear with FusedFlashLinear BEFORE FSDP wrapping
    if is_fuse_mode:
        if master_process:
            print("Replacing Linear layers with FusedFlashLinear (pre-FSDP)")
        from opacus.grad_sample.fused_flash_linear import replace_linear_with_fused
        model = replace_linear_with_fused(model)
    
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
        model = model.to(device)
    
    model.train()
    
    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # Create criterion based on model type
    if model_type == "llm":
        criterion = torch.nn.CrossEntropyLoss()
    else:  # dit
        # DiT uses MSE loss with per-sample reduction for DP
        def dit_criterion(predicted, target):
            batch_size = predicted.shape[0]
            pred_flat = predicted.reshape(batch_size, -1)
            target_flat = target.reshape(batch_size, -1)
            loss_per_element = nn.functional.mse_loss(pred_flat, target_flat, reduction='none')
            return loss_per_element.mean(dim=1)  # (B,)
        dit_criterion.reduction = "mean"
        criterion = dit_criterion
    
    # Apply DP-SGD if needed (exclude no_dp and no_dp_single)
    is_dp_mode = mode not in ["no_dp", "no_dp_single"]
    
    if is_dp_mode:
        if master_process:
            print(f"Applying DP-SGD with mode: {mode}")
        
        privacy_engine = PrivacyEngine()
        
        # Create dummy dataloader for make_private
        from torch.utils.data import TensorDataset, DataLoader
        
        if is_fsdp_mode:
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
    
    # Warmup iterations
    if master_process:
        print(f"\nRunning {warmup_iter} warmup iterations...")
    
    for i in range(warmup_iter):
        batch = generate_batch(model_type, config, device)
        
        optimizer.zero_grad()
        
        if model_type == "llm":
            if is_dp_mode:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                loss = criterion(outputs.logits, batch["labels"])
            else:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                loss = outputs.loss
        else:  # dit
            if is_dp_mode:
                predicted_noise = model(batch["images"], batch["timesteps"], batch["labels"], target_noise=None)
                if predicted_noise.shape[1] > config["in_channels"]:
                    predicted_noise = predicted_noise[:, :config["in_channels"], :, :]
                loss = criterion(predicted_noise, batch["target_noise"])
            else:
                outputs = model(batch["images"], batch["timesteps"], batch["labels"], target_noise=batch["target_noise"])
                loss = outputs["loss"]
        
        loss.backward()
        optimizer.step()
        
        if is_fsdp_mode:
            torch.cuda.synchronize()
            if dist.is_initialized():
                dist.barrier()
        
        del batch, loss
        if model_type == "llm" and not is_dp_mode:
            del outputs
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
    
    for i in range(num_iter):
        batch = generate_batch(model_type, config, device)
        
        # Time the iteration
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        
        optimizer.zero_grad()
        
        if model_type == "llm":
            if is_dp_mode:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                loss = criterion(outputs.logits, batch["labels"])
            else:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                loss = outputs.loss
        else:  # dit
            if is_dp_mode:
                predicted_noise = model(batch["images"], batch["timesteps"], batch["labels"], target_noise=None)
                if predicted_noise.shape[1] > config["in_channels"]:
                    predicted_noise = predicted_noise[:, :config["in_channels"], :, :]
                loss = criterion(predicted_noise, batch["target_noise"])
            else:
                outputs = model(batch["images"], batch["timesteps"], batch["labels"], target_noise=batch["target_noise"])
                loss = outputs["loss"]
        
        loss.backward()
        optimizer.step()
        
        end_event.record()
        torch.cuda.synchronize()
        
        if is_fsdp_mode and dist.is_initialized():
            dist.barrier()
        
        iter_time = start_event.elapsed_time(end_event)
        total_time += iter_time
        
        if master_process:
            print(f"  Iteration {i+1}/{num_iter}: {iter_time:.2f} ms")
        
        del batch, loss
        if model_type == "llm" and not is_dp_mode:
            del outputs
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
    
    # Synchronize before storing results
    if is_fsdp_mode and dist.is_initialized():
        dist.barrier()
    
    # Store results in shared dict (only from rank 0)
    if master_process:
        results_dict["peak_memory_mb"] = peak_memory_mb
        results_dict["peak_memory_gb"] = peak_memory_gb
        results_dict["avg_time_ms"] = avg_time_ms
    
    # Final synchronization before cleanup
    if is_fsdp_mode and dist.is_initialized():
        dist.barrier()
    
    # Only cleanup distributed for FSDP modes
    if is_fsdp_mode:
        cleanup()


def run_experiment(model_type, config):
    """Run the experiment with the given configuration"""
    mode = config["mode"]
    
    # Determine if this is an FSDP mode (multi-GPU)
    is_fsdp_mode = mode in ["ghost_fsdp", "flash_fsdp", "flash_fsdp_bk", "ghost_fsdp_bk", "flash_fsdp_fuse", "flash_fsdp_fuse_bk", "no_dp"]
    
    if is_fsdp_mode:
        world_size = torch.cuda.device_count()
        if world_size == 0:
            raise RuntimeError("No CUDA devices available")
    else:
        world_size = 1
        if torch.cuda.device_count() == 0:
            raise RuntimeError("No CUDA devices available")
    
    # Compute seq_length for display purposes
    if model_type == "llm":
        seq_length = config["seq_length"]
    else:  # dit
        seq_length = (config["image_size"] // config["patch_size"]) ** 2
    
    print(f"\n{'#'*80}")
    print(f"Configuration:")
    print(f"  Model Type: {model_type}")
    print(f"  Mode: {mode}")
    if model_type == "llm":
        print(f"  Model: {config['model_name']}")
        print(f"  Sequence Length: {seq_length}")
    else:
        print(f"  Image Size: {config['image_size']}")
        print(f"  Patch Size: {config['patch_size']}")
        print(f"  Num Tokens: {seq_length}")
        print(f"  Hidden Dim: {config['hidden_dim']}")
        print(f"  Num Layers: {config['num_layers']}")
    print(f"  Batch Size (per GPU): {config['batch_size']}")
    print(f"  Number of GPUs: {world_size}")
    print(f"  Total Batch Size: {config['batch_size'] * world_size}")
    print(f"  Profiling Iterations: {config['num_iter']}")
    print(f"  Warmup Iterations: {config['warmup_iter']}")
    print(f"{'#'*80}\n")
    
    # Use multiprocessing manager for sharing results
    manager = mp.Manager()
    results_dict = manager.dict()
    
    if is_fsdp_mode:
        mp.spawn(
            run_experiment_worker,
            args=(
                world_size,
                model_type,
                mode,
                config,
                config["num_iter"],
                config["warmup_iter"],
                config["learning_rate"],
                config["sigma"],
                config["max_grad_norm"],
                results_dict,
            ),
            nprocs=world_size,
            join=True,
        )
    else:
        run_experiment_worker(
            rank=0,
            world_size=1,
            model_type=model_type,
            mode=mode,
            config=config,
            num_iter=config["num_iter"],
            warmup_iter=config["warmup_iter"],
            learning_rate=config["learning_rate"],
            sigma=config["sigma"],
            max_grad_norm=config["max_grad_norm"],
            results_dict=results_dict,
        )
    
    # Extract results
    results = {
        "model_type": model_type,
        "mode": mode,
        "seq_length": seq_length,
        "batch_size": config["batch_size"],
        "total_batch_size": config["batch_size"] * world_size,
        "num_gpus": world_size,
        "peak_memory_mb": results_dict.get("peak_memory_mb", 0),
        "peak_memory_gb": results_dict.get("peak_memory_gb", 0),
        "avg_time_ms": results_dict.get("avg_time_ms", 0),
        "config": config,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run single FSDP/Single-GPU profiling experiment")
    
    # Common arguments
    parser.add_argument("--model-type", type=str, required=True,
                       choices=["llm", "dit"],
                       help="Model type: llm (HuggingFace LLM) or dit (DiT)")
    parser.add_argument("--mode", type=str, required=True,
                       choices=["no_dp", "no_dp_single", "ghost_fsdp", "flash_fsdp", "flash_fsdp_bk", "ghost_fsdp_bk", "flash_fsdp_fuse", "flash_fsdp_fuse_bk", "flash", "flash_bk", "ghost", "ghost_bk", "flash_fuse", "flash_fuse_bk"],
                       help="Training mode")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size per GPU")
    parser.add_argument("--num-iter", type=int, default=3,
                       help="Number of profiling iterations")
    parser.add_argument("--warmup-iter", type=int, default=1,
                       help="Number of warmup iterations")
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSON file path")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--sigma", type=float, default=1.0,
                       help="Noise multiplier for DP")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                       help="Max gradient norm for DP")
    
    # LLM-specific arguments
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                       help="HuggingFace model name (for LLM)")
    parser.add_argument("--token", type=str, default="",
                       help="HuggingFace token (for LLM)")
    parser.add_argument("--vocab-size", type=int, default=128256,
                       help="Vocabulary size for synthetic data (for LLM)")
    parser.add_argument("--seq-length", type=int, default=1024,
                       help="Sequence length (for LLM)")
    
    # DiT-specific arguments
    parser.add_argument("--image-size", type=int, default=256,
                       help="Image size (for DiT)")
    parser.add_argument("--patch-size", type=int, default=8,
                       help="Patch size (for DiT)")
    parser.add_argument("--in-channels", type=int, default=4,
                       help="Input channels (for DiT)")
    parser.add_argument("--hidden-dim", type=int, default=1152,
                       help="Hidden dimension (for DiT)")
    parser.add_argument("--num-layers", type=int, default=28,
                       help="Number of layers (for DiT)")
    parser.add_argument("--num-heads", type=int, default=16,
                       help="Number of attention heads (for DiT)")
    parser.add_argument("--num-classes", type=int, default=1000,
                       help="Number of classes (for DiT)")
    
    args = parser.parse_args()
    
    # Build config based on model type
    config = {
        "mode": args.mode,
        "batch_size": args.batch_size,
        "num_iter": args.num_iter,
        "warmup_iter": args.warmup_iter,
        "learning_rate": args.learning_rate,
        "sigma": args.sigma,
        "max_grad_norm": args.max_grad_norm,
    }
    
    if args.model_type == "llm":
        config.update({
            "model_name": args.model_name,
            "token": args.token,
            "vocab_size": args.vocab_size,
            "seq_length": args.seq_length,
        })
    else:  # dit
        config.update({
            "image_size": args.image_size,
            "patch_size": args.patch_size,
            "in_channels": args.in_channels,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "num_classes": args.num_classes,
        })
    
    # Run experiment
    aggressive_cleanup()
    results = run_experiment(args.model_type, config)
    aggressive_cleanup()
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ Experiment completed!")
    print(f"✅ Model Type: {results['model_type']}")
    print(f"✅ Mode: {results['mode']}")
    print(f"✅ Seq Length: {results['seq_length']}")
    print(f"✅ Peak Memory: {results['peak_memory_gb']:.2f} GB")
    print(f"✅ Avg Time: {results['avg_time_ms']:.2f} ms")
    print(f"✅ Results saved to: {args.output}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
