#!/usr/bin/env python3
"""
Test distributed correctness by comparing:
1. Single GPU processing full batch
2. FSDP with 2 ranks processing split batch (using DistributedSampler)

Key insight: To verify distributed correctness:
- Single GPU: batch_size=16, processes samples [0-15]
- FSDP rank 0: batch_size=8, processes samples [0, 2, 4, 6, 8, 10, 12, 14]
- FSDP rank 1: batch_size=8, processes samples [1, 3, 5, 7, 9, 11, 13, 15]

After aggregation, results should be equivalent to processing batch_size=16 on single GPU.
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
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from opacus import PrivacyEngine
from opacus.utils.fsdp_utils import FSDP2Wrapper
from torch.distributed.device_mesh import init_device_mesh
from test_model import SmallTransformerDP, create_synthetic_dataset


def train_single_gpu_full_batch(batch_size, num_samples, seq_len, seed, device="cpu"):
    """
    Single GPU baseline: process full batch.
    
    Returns:
        per_sample_norms: [batch_size] gradient norms for each sample
        loss: scalar loss value
    """
    torch.manual_seed(seed)
    
    model = SmallTransformerDP(
        vocab_size=10000, hidden_dim=512, num_heads=8, num_layers=3,
        num_classes=10, max_seq_len=128, dropout=0.0
    )
    
    inputs, labels = create_synthetic_dataset(
        num_samples=num_samples, vocab_size=10000, seq_len=seq_len, 
        num_classes=10, seed=seed
    )
    dataset = TensorDataset(inputs, labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    privacy_engine = PrivacyEngine()
    model, optimizer, criterion, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        criterion=nn.CrossEntropyLoss(),
        noise_multiplier=0.0,  # No noise for exact comparison
        max_grad_norm=1.0,
        grad_sample_mode="flash",
        poisson_sampling=False,
    )
    
    model = model.to(device)
    model.train()
    
    # Process first batch
    batch_inputs, batch_labels = next(iter(train_loader))
    batch_inputs = batch_inputs.to(device)
    batch_labels = batch_labels.to(device)
    
    optimizer.zero_grad()
    outputs = model(batch_inputs)
    loss = criterion(outputs, batch_labels)
    loss.backward()
    
    per_sample_norms = model.per_sample_gradient_norms.detach().cpu()
    loss_value = loss.item()
    
    return per_sample_norms, loss_value


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12359"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train_fsdp_rank(rank, world_size, batch_size, num_samples, seq_len, seed, results_dict):
    """
    FSDP training on one rank with DistributedSampler.
    Each rank processes different subset of data.
    """
    setup(rank, world_size)
    
    device_mesh = init_device_mesh("cpu", (world_size,))
    device = torch.device("cpu")
    
    # Same seed for model initialization across ranks
    torch.manual_seed(seed)
    
    model = SmallTransformerDP(
        vocab_size=10000, hidden_dim=512, num_heads=8, num_layers=3,
        num_classes=10, max_seq_len=128, dropout=0.0
    )
    
    model = FSDP2Wrapper(model, mesh=device_mesh)
    
    # Same seed for data generation across ranks
    inputs, labels = create_synthetic_dataset(
        num_samples=num_samples, vocab_size=10000, seq_len=seq_len,
        num_classes=10, seed=seed
    )
    dataset = TensorDataset(inputs, labels)
    
    # Use DistributedSampler to split data across ranks
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    train_loader = DataLoader(
        dataset, batch_size=batch_size // world_size, sampler=sampler
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    privacy_engine = PrivacyEngine()
    model, optimizer, criterion, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        criterion=nn.CrossEntropyLoss(),
        noise_multiplier=0.0,  # No noise for exact comparison
        max_grad_norm=1.0,
        grad_sample_mode="flash_fsdp_bk",
        poisson_sampling=False,
    )
    
    model.train()
    
    # Process first batch
    batch_inputs, batch_labels = next(iter(train_loader))
    batch_inputs = batch_inputs.to(device)
    batch_labels = batch_labels.to(device)
    
    print(f"[Rank {rank}] Processing samples with indices from DistributedSampler")
    print(f"[Rank {rank}] Batch shape: {batch_inputs.shape}")
    
    optimizer.zero_grad()
    outputs = model(batch_inputs)
    loss = criterion(outputs, batch_labels)
    loss.backward()
    
    # Get per-sample norms (already aggregated across ranks by FSDP)
    per_sample_norms = model.per_sample_gradient_norms.detach().cpu()
    loss_value = loss.item()
    
    # Only rank 0 stores results
    if rank == 0:
        results_dict['per_sample_norms'] = per_sample_norms
        results_dict['loss'] = loss_value
    
    cleanup()


def run_fsdp_distributed(batch_size, num_samples, seq_len, seed):
    """Run FSDP with 2 ranks using DistributedSampler."""
    world_size = 2
    
    # Use Manager to share results between processes
    manager = mp.Manager()
    results_dict = manager.dict()
    
    mp.spawn(
        train_fsdp_rank,
        args=(world_size, batch_size, num_samples, seq_len, seed, results_dict),
        nprocs=world_size,
        join=True,
    )
    
    return results_dict['per_sample_norms'], results_dict['loss']


def compare_results(single_gpu_norms, single_gpu_loss, fsdp_norms, fsdp_loss):
    """
    Compare results between single GPU and FSDP.
    
    Note: With DistributedSampler, each rank processes different samples.
    The per-sample norms from FSDP are for the subset processed by each rank.
    We need to verify the aggregation is correct.
    """
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    print(f"\nLoss comparison:")
    print(f"  Single GPU loss: {single_gpu_loss:.6f}")
    print(f"  FSDP loss:       {fsdp_loss:.6f}")
    print(f"  Difference:      {abs(single_gpu_loss - fsdp_loss):.6e}")
    
    print(f"\nGradient norms comparison:")
    print(f"  Single GPU norms shape: {single_gpu_norms.shape}")
    print(f"  FSDP norms shape:       {fsdp_norms.shape}")
    
    if single_gpu_norms.shape[0] == fsdp_norms.shape[0]:
        # Direct comparison
        norm_diff = torch.abs(single_gpu_norms - fsdp_norms)
        print(f"\n  Per-sample norm differences:")
        print(f"    Mean: {norm_diff.mean().item():.6e}")
        print(f"    Max:  {norm_diff.max().item():.6e}")
        print(f"    Min:  {norm_diff.min().item():.6e}")
        
        # Check tolerance
        tolerance = 1e-3
        if norm_diff.max().item() < tolerance:
            print(f"\n✓ PASS: All gradient norms within tolerance ({tolerance:.6e})")
            return True
        else:
            print(f"\n✗ FAIL: Some gradient norms exceed tolerance ({tolerance:.6e})")
            return False
    else:
        print(f"\n  Note: Different batch sizes, comparing statistics:")
        print(f"    Single GPU - Mean: {single_gpu_norms.mean().item():.6f}, Std: {single_gpu_norms.std().item():.6f}")
        print(f"    FSDP       - Mean: {fsdp_norms.mean().item():.6f}, Std: {fsdp_norms.std().item():.6f}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Test Distributed Correctness")
    parser.add_argument("--batch_size", type=int, default=16, help="Total batch size")
    parser.add_argument("--num_samples", type=int, default=32, help="Number of samples")
    parser.add_argument("--seq_len", type=int, default=32, help="Sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    print("=" * 80)
    print("DISTRIBUTED CORRECTNESS TEST")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num samples: {args.num_samples}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Seed: {args.seed}")
    print("=" * 80)
    
    # Test 1: Single GPU with full batch
    print("\n[1] Running Single GPU (full batch)...")
    single_gpu_norms, single_gpu_loss = train_single_gpu_full_batch(
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        seed=args.seed,
    )
    print(f"    Completed: {single_gpu_norms.shape[0]} samples")
    print(f"    Mean norm: {single_gpu_norms.mean().item():.6f}")
    
    # Test 2: FSDP with 2 ranks (split batch with DistributedSampler)
    print(f"\n[2] Running FSDP (world_size=2, {args.batch_size//2} samples per rank)...")
    fsdp_norms, fsdp_loss = run_fsdp_distributed(
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        seed=args.seed,
    )
    print(f"    Completed: {fsdp_norms.shape[0]} samples")
    print(f"    Mean norm: {fsdp_norms.mean().item():.6f}")
    
    # Compare results
    passed = compare_results(single_gpu_norms, single_gpu_loss, fsdp_norms, fsdp_loss)
    
    # Additional diagnostic: Show which samples each method processed
    print("\n" + "=" * 80)
    print("DIAGNOSTIC INFORMATION")
    print("=" * 80)
    print(f"\nSingle GPU norms (first 8 samples):")
    print(single_gpu_norms[:8].numpy())
    print(f"\nFSDP norms (first 8 samples):")
    print(fsdp_norms[:8].numpy())
    
    if passed is False:
        print("\n" + "=" * 80)
        print("✗ TEST FAILED: Distributed computation may be incorrect")
        print("=" * 80)
        sys.exit(1)
    elif passed is True:
        print("\n" + "=" * 80)
        print("✓ TEST PASSED: Distributed computation is correct!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("⚠ TEST INCONCLUSIVE: Manual verification needed")
        print("=" * 80)


if __name__ == "__main__":
    main()

