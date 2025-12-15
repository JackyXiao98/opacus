#!/usr/bin/env python3
"""
Run a single DiT profiling experiment using flashnorm for DP/grad-sample handling.
Supports single-GPU and FSDP modes.
"""

import argparse
import gc
import json
import os
import statistics
import sys

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from flashnorm import PrivacyEngine
from flashnorm.utils.fsdp_utils import FSDP2Wrapper

# Add parent directories to path for DiT model imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..")
)


def aggressive_cleanup():
    """Aggressive memory cleanup."""
    for _ in range(3):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    gc.collect(generation=2)


def create_dit_model(config, device, master_process=True):
    """Create DiT model (same as dp-train.py)."""
    from dit_models import DiT_models

    model_name = config.get("model_name", "DiT-XL/2")
    input_size = config["image_size"] // 8  # Latent space size after VAE

    if master_process:
        print(f"Creating DiT model: {model_name}, input_size={input_size}")

    model = DiT_models[model_name](
        input_size=input_size,
        num_classes=config["num_classes"],
    )

    return model


def generate_dit_batch(config, device):
    """Generate synthetic batch for DiT (latent space)."""
    batch_size = config["batch_size"]
    in_channels = config["in_channels"]
    image_size = config["image_size"]
    num_classes = config["num_classes"]

    latent_size = image_size // 8
    latents = torch.randn(batch_size, in_channels, latent_size, latent_size, device=device)
    timesteps = torch.randint(0, 1000, (batch_size,), device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    target_noise = torch.randn_like(latents)

    return {
        "images": latents,
        "timesteps": timesteps,
        "labels": labels,
        "target_noise": target_noise,
    }


def setup(rank, world_size):
    """Setup distributed training."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """Cleanup distributed training."""
    dist.destroy_process_group()


def run_experiment_worker(
    rank,
    world_size,
    mode,
    config,
    num_iter,
    warmup_iter,
    learning_rate,
    sigma,
    max_grad_norm,
    results_dict,
):
    """Run experiment on a single GPU worker."""
    torch.cuda.set_device(rank)
    time_stat = config.get("time_stat", "median")

    is_fsdp_mode = mode in [
        "ghost_fsdp",
        "flash_fsdp",
        "flash_fsdp_bk",
        "ghost_fsdp_bk",
        "flash_fsdp_fuse",
        "flash_fsdp_fuse_bk",
        "no_dp",
    ]

    if is_fsdp_mode:
        setup(rank, world_size)
        dist.barrier()

    master_process = rank == 0
    torch.manual_seed(1337 + rank)

    model_name = config.get("model_name", "DiT-XL/2")
    patch_size = int(model_name.split("/")[-1])
    latent_size = config.get("image_size", 256) // 8
    seq_length = config.get("seq_length")
    if seq_length is None:
        seq_length = (latent_size // patch_size) ** 2
    batch_size = config["batch_size"]

    if master_process:
        print(f"\n{'='*80}")
        print(
            f"EXPERIMENT: mode={mode}, seq_length={seq_length}, batch_size={batch_size}, image_size={config['image_size']}"
        )
        print(f"{'='*80}\n")

    device = torch.device(f"cuda:{rank}")

    model = create_dit_model(config, device, master_process)

    is_fuse_mode = mode in [
        "flash_fsdp_fuse",
        "flash_fsdp_fuse_bk",
        "flash_fuse",
        "flash_fuse_bk",
    ]

    if is_fuse_mode:
        if master_process:
            print("Replacing Linear layers with FusedFlashLinear (pre-FSDP)")
        from flashnorm.grad_sample.fused_flash_linear import replace_linear_with_fused

        model = replace_linear_with_fused(model)

    if is_fsdp_mode:
        if master_process:
            print("Wrapping model with FSDP2")
        mp_policy = dist.fsdp.MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
        model = FSDP2Wrapper(model, mp_policy=mp_policy)
    else:
        if master_process:
            print("Using single-GPU mode (no FSDP)")
        model = model.to(device)

    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    def dit_criterion(predicted, target):
        bsz = predicted.shape[0]
        pred_flat = predicted.reshape(bsz, -1)
        target_flat = target.reshape(bsz, -1)
        loss_per_element = nn.functional.mse_loss(pred_flat, target_flat, reduction="none")
        return loss_per_element.mean(dim=1)

    dit_criterion.reduction = "mean"
    criterion = dit_criterion

    is_dp_mode = mode not in ["no_dp", "no_dp_single"]

    if is_dp_mode:
        if master_process:
            print(f"Applying DP-SGD with mode: {mode}")

        privacy_engine = PrivacyEngine()

        from torch.utils.data import DataLoader, TensorDataset

        if is_fsdp_mode:
            from torch.utils.data import DistributedSampler

            dummy_data = torch.randn(batch_size * world_size, seq_length)
            dummy_labels = torch.randint(0, 3, (batch_size * world_size,))
            dummy_dataset = TensorDataset(dummy_data, dummy_labels)
            dummy_dataloader = DataLoader(
                dummy_dataset,
                batch_size=batch_size,
                sampler=DistributedSampler(
                    dummy_dataset, num_replicas=world_size, rank=rank
                ),
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

    if master_process:
        print(f"\nRunning {warmup_iter} warmup iterations...")

    for i in range(warmup_iter):
        batch = generate_dit_batch(config, device)

        optimizer.zero_grad()

        predicted_noise = model(
            batch["images"], batch["timesteps"], batch["labels"]
        )
        if predicted_noise.shape[1] > config["in_channels"]:
            predicted_noise = predicted_noise[:, : config["in_channels"], :, :]
        if is_dp_mode:
            loss = criterion(predicted_noise, batch["target_noise"])
        else:
            loss = torch.nn.functional.mse_loss(
                predicted_noise, batch["target_noise"]
            )

        loss.backward()
        optimizer.step()

        if is_fsdp_mode:
            torch.cuda.synchronize()
            if dist.is_initialized():
                dist.barrier()

        del batch, loss
        torch.cuda.empty_cache()

    if is_fsdp_mode and dist.is_initialized():
        dist.barrier()

    torch.cuda.reset_peak_memory_stats(device)

    if master_process:
        print(f"\nRunning {num_iter} profiling iterations...")

    iter_times = []

    for i in range(num_iter):
        batch = generate_dit_batch(config, device)

        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        optimizer.zero_grad()

        predicted_noise = model(
            batch["images"], batch["timesteps"], batch["labels"]
        )
        if predicted_noise.shape[1] > config["in_channels"]:
            predicted_noise = predicted_noise[:, : config["in_channels"], :, :]
        if is_dp_mode:
            loss = criterion(predicted_noise, batch["target_noise"])
        else:
            loss = torch.nn.functional.mse_loss(
                predicted_noise, batch["target_noise"]
            )

        loss.backward()
        optimizer.step()

        end_event.record()
        torch.cuda.synchronize()

        if is_fsdp_mode and dist.is_initialized():
            dist.barrier()

        iter_time = start_event.elapsed_time(end_event)
        iter_times.append(iter_time)

        if master_process:
            print(f"  Iteration {i+1}/{num_iter}: {iter_time:.2f} ms")

        del batch, loss
        torch.cuda.empty_cache()

    if time_stat == "mean":
        time_value_ms = sum(iter_times) / len(iter_times)
    elif time_stat == "median":
        time_value_ms = statistics.median(iter_times)
    else:
        raise ValueError(f"Unknown time_stat: {time_stat}")
    peak_memory_bytes = torch.cuda.max_memory_allocated(device)
    peak_memory_mb = peak_memory_bytes / (1024**2)
    peak_memory_gb = peak_memory_bytes / (1024**3)

    if is_fsdp_mode and dist.is_initialized():
        dist.barrier()

    if master_process:
        print(f"\n⏱️  Iteration time ({time_stat}): {time_value_ms:.2f} ms")
        print(f"💾 Peak memory usage: {peak_memory_gb:.2f} GB ({peak_memory_mb:.2f} MB)")

    if is_fsdp_mode and dist.is_initialized():
        dist.barrier()

    if master_process:
        results_dict["peak_memory_mb"] = peak_memory_mb
        results_dict["peak_memory_gb"] = peak_memory_gb
        results_dict["avg_time_ms"] = time_value_ms
        results_dict["time_stat"] = time_stat

    if is_fsdp_mode:
        cleanup()


def run_experiment(config):
    """Run the experiment with the given configuration."""
    mode = config["mode"]

    is_fsdp_mode = mode in [
        "ghost_fsdp",
        "flash_fsdp",
        "flash_fsdp_bk",
        "ghost_fsdp_bk",
        "flash_fsdp_fuse",
        "flash_fsdp_fuse_bk",
        "no_dp",
    ]

    if is_fsdp_mode:
        world_size = torch.cuda.device_count()
        if world_size == 0:
            raise RuntimeError("No CUDA devices available")
    else:
        world_size = 1
        if torch.cuda.device_count() == 0:
            raise RuntimeError("No CUDA devices available")

    model_name = config.get("model_name", "DiT-XL/2")
    patch_size = int(model_name.split("/")[-1])
    latent_size = config["image_size"] // 8
    seq_length = config.get("seq_length")
    if seq_length is None:
        seq_length = (latent_size // patch_size) ** 2

    print(f"\n{'#'*80}")
    print("Configuration:")
    print(f"  Mode: {mode}")
    print(f"  DiT Model: {config['model_name']}")
    print(f"  Image Size: {config['image_size']} (latent: {latent_size})")
    print(f"  Patch Size: {patch_size}")
    print(f"  Num Tokens: {seq_length}")
    print(f"  Batch Size (per GPU): {config['batch_size']}")
    print(f"  Number of GPUs: {world_size}")
    print(f"  Total Batch Size: {config['batch_size'] * world_size}")
    print(f"  Profiling Iterations: {config['num_iter']}")
    print(f"  Warmup Iterations: {config['warmup_iter']}")
    print(f"{'#'*80}\n")

    manager = mp.Manager()
    results_dict = manager.dict()

    if is_fsdp_mode:
        mp.spawn(
            run_experiment_worker,
            args=(
                world_size,
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
            mode=mode,
            config=config,
            num_iter=config["num_iter"],
            warmup_iter=config["warmup_iter"],
            learning_rate=config["learning_rate"],
            sigma=config["sigma"],
            max_grad_norm=config["max_grad_norm"],
            results_dict=results_dict,
        )

    results = {
        "model_type": "dit",
        "mode": mode,
        "seq_length": seq_length,
        "batch_size": config["batch_size"],
        "total_batch_size": config["batch_size"] * world_size,
        "num_gpus": world_size,
        "peak_memory_mb": results_dict.get("peak_memory_mb", 0),
        "peak_memory_gb": results_dict.get("peak_memory_gb", 0),
        "avg_time_ms": results_dict.get("avg_time_ms", 0),
        "time_stat": results_dict.get("time_stat", config.get("time_stat", "median")),
        "config": config,
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run single DiT profiling experiment with flashnorm"
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "no_dp",
            "no_dp_single",
            "ghost_fsdp",
            "flash_fsdp",
            "flash_fsdp_bk",
            "ghost_fsdp_bk",
            "flash_fsdp_fuse",
            "flash_fsdp_fuse_bk",
            "flash",
            "flash_bk",
            "ghost",
            "ghost_bk",
            "flash_fuse",
            "flash_fuse_bk",
        ],
        help="Training mode",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--num-iter",
        type=int,
        default=3,
        help="Number of profiling iterations",
    )
    parser.add_argument(
        "--warmup-iter",
        type=int,
        default=1,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Noise multiplier for DP",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Max gradient norm for DP",
    )
    parser.add_argument(
        "--time-stat",
        type=str,
        default="median",
        choices=["mean", "median"],
        help="Statistic for iteration time reporting",
    )

    parser.add_argument(
        "--dit-model-name",
        type=str,
        default="DiT-XL/2",
        choices=[
            "DiT-XL/2",
            "DiT-XL/4",
            "DiT-XL/8",
            "DiT-L/2",
            "DiT-L/4",
            "DiT-L/8",
            "DiT-B/2",
            "DiT-B/4",
            "DiT-B/8",
            "DiT-S/2",
            "DiT-S/4",
            "DiT-S/8",
        ],
        help="DiT model variant",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Image size (latent_size = image_size // 8)",
    )
    parser.add_argument(
        "--in-channels",
        type=int,
        default=4,
        help="Input channels (for DiT)",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=None,
        help="Optional sequence length override (tokens), defaults to derived from image_size and model",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
        help="Number of classes (for DiT)",
    )

    args = parser.parse_args()

    config = {
        "mode": args.mode,
        "batch_size": args.batch_size,
        "num_iter": args.num_iter,
        "warmup_iter": args.warmup_iter,
        "learning_rate": args.learning_rate,
        "sigma": args.sigma,
        "max_grad_norm": args.max_grad_norm,
        "model_name": args.dit_model_name,
        "image_size": args.image_size,
        "in_channels": args.in_channels,
        "num_classes": args.num_classes,
        "time_stat": args.time_stat,
    }
    if args.seq_length is not None:
        config["seq_length"] = args.seq_length

    aggressive_cleanup()
    results = run_experiment(config)
    aggressive_cleanup()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ Experiment completed!")
    print(f"✅ Mode: {results['mode']}")
    print(f"✅ Seq Length: {results['seq_length']}")
    print(f"✅ Peak Memory: {results['peak_memory_gb']:.2f} GB")
    print(f"✅ Time ({results['time_stat']}): {results['avg_time_ms']:.2f} ms")
    print(f"✅ Results saved to: {args.output}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

