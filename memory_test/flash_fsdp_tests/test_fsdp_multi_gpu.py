#!/usr/bin/env python3
"""
FSDP multi-GPU training with Flash Clipping.
This tests flash_fsdp mode and compares results with single GPU baseline.
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader, TensorDataset

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from opacus import PrivacyEngine
from opacus.utils.fsdp_utils import FSDP2Wrapper
from test_model import SmallTransformerDP, create_synthetic_dataset


def setup(rank, world_size):
    """Initialize distributed training."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", rank=rank, world_size=world_size)


def cleanup():
    """Clean up distributed training."""
    dist.destroy_process_group()


def train_fsdp(
    rank,
    world_size,
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    num_epochs=1,
    save_dir="./results_fsdp",
):
    """
    Train model with FSDP and track metrics.
    
    Returns:
        metrics: Dictionary containing training metrics (only on rank 0)
    """
    model.train()
    
    metrics = {
        "losses": [],
        "step_times": [],
        "grad_norms": [],
        "param_norms": [],
    }
    
    step = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            step_start = time.time()
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass (criterion.backward() for ghost/flash modes)
            loss.backward()
            
            # Track gradient norm before optimizer step (only on rank 0)
            if rank == 0:
                total_grad_norm = 0.0
                if hasattr(model, 'per_sample_gradient_norms'):
                    try:
                        # Get per-sample gradient norms from the module
                        per_sample_norms = model.per_sample_gradient_norms
                        # Average norm across the batch
                        total_grad_norm = float(per_sample_norms.mean().item())
                    except:
                        total_grad_norm = 0.0
                metrics["grad_norms"].append(total_grad_norm)
            
            optimizer.step()
            
            step_time = time.time() - step_start
            
            # FIXED: Compute GLOBAL parameter norm (all ranks must participate in all_reduce)
            local_squared_norm = 0.0
            for p in model.parameters():
                local_squared_norm += p.data.norm(2).item() ** 2
            
            # All ranks must participate in collective operations!
            global_squared_norm_tensor = torch.tensor(local_squared_norm, device=device)
            dist.all_reduce(global_squared_norm_tensor, op=dist.ReduceOp.SUM)
            total_param_norm = global_squared_norm_tensor.item() ** 0.5
            
            # Track metrics on rank 0 only
            if rank == 0:
                # Record metrics
                metrics["losses"].append(loss.item())
                metrics["step_times"].append(step_time)
                metrics["param_norms"].append(total_param_norm)
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Step {step+1}, "
                          f"Loss: {loss.item():.4f}, "
                          f"Grad Norm: {total_grad_norm:.4f}")
            
            epoch_loss += loss.item()
            step += 1
        
        if rank == 0:
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")
    
    # Save metrics and checkpoint on rank 0
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        torch.save(model.state_dict(), os.path.join(save_dir, "model_checkpoint.pt"))
        print(f"Results saved to {save_dir}")
    
    return metrics if rank == 0 else None


def run_training(rank, world_size, args):
    """Main training function for each process."""
    setup(rank, world_size)
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        device_type = "cuda"
    else:
        device = torch.device("cpu")
        device_type = "cpu"
    
    # Initialize device mesh for FSDP
    device_mesh = init_device_mesh(device_type, (world_size,))
    
    # Set random seed (same across all ranks for reproducibility)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    if rank == 0:
        print("=" * 80)
        print("FSDP Flash Clipping Training")
        print("=" * 80)
        print(f"World size: {world_size}")
        print(f"Device: {device}")
        print(f"Batch size (per GPU): {args.batch_size // world_size}")
        print(f"Total batch size: {args.batch_size}")
        print(f"Num samples: {args.num_samples}")
        print(f"Sequence length: {args.seq_len}")
        print(f"Learning rate: {args.lr}")
        print(f"Noise multiplier: {args.noise_multiplier}")
        print(f"Max grad norm: {args.max_grad_norm}")
        print(f"Seed: {args.seed}")
        print("=" * 80)
    
    # Create model
    model = SmallTransformerDP(
        vocab_size=10000,
        hidden_dim=512,
        num_heads=8,
        num_layers=3,
        num_classes=10,
        max_seq_len=128,
        dropout=0.0,  # No dropout for deterministic results
    )
    
    if rank == 0:
        print(f"Model parameters: {model.count_parameters():,}")
    
    # Wrap with FSDP
    model = FSDP2Wrapper(model, mesh=device_mesh)
    
    # Create dataset
    inputs, labels = create_synthetic_dataset(
        num_samples=args.num_samples,
        vocab_size=10000,
        seq_len=args.seq_len,
        num_classes=10,
        seed=args.seed,
    )
    dataset = TensorDataset(inputs, labels)
    
    # FIXED: Replicate full dataset on all ranks to match single GPU behavior
    # Each rank processes the SAME batch (not partitioned), just like single GPU
    # This ensures identical forward passes and loss values across all ranks
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Initialize Privacy Engine
    privacy_engine = PrivacyEngine()
    model, optimizer, criterion, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        criterion=nn.CrossEntropyLoss(),
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
        grad_sample_mode="flash_fsdp_bk",  # Use flash_fsdp mode
        poisson_sampling=False,  # Disable for deterministic results
    )
    
    if rank == 0:
        print(f"Privacy Engine initialized with grad_sample_mode='flash_fsdp'")
    
    # Train
    metrics = train_fsdp(
        rank=rank,
        world_size=world_size,
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir,
    )
    
    # Print summary on rank 0
    if rank == 0 and metrics is not None:
        print("\n" + "=" * 80)
        print("Training Summary")
        print("=" * 80)
        print(f"Total steps: {len(metrics['losses'])}")
        print(f"Final loss: {metrics['losses'][-1]:.6f}")
        print(f"Average step time: {sum(metrics['step_times']) / len(metrics['step_times']):.4f}s")
        print(f"Final grad norm: {metrics['grad_norms'][-1]:.6f}")
        print(f"Final param norm: {metrics['param_norms'][-1]:.6f}")
        
        # Get privacy budget
        epsilon = privacy_engine.get_epsilon(delta=1e-5)
        print(f"Privacy budget (ε): {epsilon:.2f} at δ=1e-5")
        print("=" * 80)
    
    cleanup()


def main():
    parser = argparse.ArgumentParser(description="FSDP Flash Clipping Training")
    parser.add_argument("--batch_size", type=int, default=32, help="Total batch size across all GPUs")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--noise_multiplier", type=float, default=1.0, help="DP noise multiplier")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="./results_fsdp", help="Save directory")
    parser.add_argument("--world_size", type=int, default=None, help="Number of GPUs (default: all available)")
    args = parser.parse_args()
    
    # Determine world size
    if args.world_size is None:
        args.world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    if args.world_size > 1 and torch.cuda.is_available():
        # Multi-GPU training
        mp.spawn(
            run_training,
            args=(args.world_size, args),
            nprocs=args.world_size,
            join=True,
        )
    else:
        # Single GPU FSDP training
        print("Running with world_size=1 (single GPU FSDP)")
        run_training(0, 1, args)


if __name__ == "__main__":
    main()

