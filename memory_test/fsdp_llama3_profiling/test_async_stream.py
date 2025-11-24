#!/usr/bin/env python3
"""
Test script for async CUDA stream norm computation optimization.

This test verifies:
1. Correctness: Async and sync computation produce identical norms
2. Performance: Async computation is faster by overlapping with FSDP communication
3. No deadlock: Works correctly with Ghost Clipping + FSDP (two-pass backward)

Async Stream Architecture:
- Norm computation runs in parallel CUDA stream
- Overlaps with FSDP backward pass communication
- Eliminates pipeline bubbles and blocking

Usage:
    # Test correctness (single-GPU)
    python test_async_stream.py --mode correctness
    
    # Test performance (requires 2 GPUs for FSDP)
    torchrun --nproc_per_node=2 test_async_stream.py --mode performance
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
from opacus import PrivacyEngine


class DPMultiheadAttentionWithFlashAttention(nn.Module):
    """
    DP-compatible MultiheadAttention using PyTorch's scaled_dot_product_attention (Flash Attention).
    This uses F.scaled_dot_product_attention when available (PyTorch 2.0+).
    """
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        batch_first=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Separate linear layers for Q, K, V to allow DP grad sample computation
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout_module = nn.Dropout(dropout)
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False):
        # Get batch size and sequence length
        if self.batch_first:
            bsz, tgt_len, _ = query.shape
            _, src_len, _ = key.shape
        else:
            tgt_len, bsz, _ = query.shape
            src_len, bsz, _ = key.shape
            # Convert to batch_first for easier processing
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention: (B, L, D) -> (B, L, H, D_head) -> (B, H, L, D_head)
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Use PyTorch's scaled_dot_product_attention (Flash Attention)
        # This will automatically use Flash Attention if available
        if hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ with Flash Attention support
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False
            )
        else:
            # Fallback to manual attention computation
            scaling = float(self.head_dim) ** -0.5
            attn_weights = torch.matmul(q * scaling, k.transpose(-2, -1))
            
            if attn_mask is not None:
                attn_weights = attn_weights + attn_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout_module(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back: (B, H, L, D_head) -> (B, L, H, D_head) -> (B, L, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        # Convert back to original format if needed
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        if need_weights:
            # For compatibility, return None for weights when using Flash Attention
            return output, None
        else:
            return output, None


class DPCompatibleTransformerLayerWithFlashAttention(nn.Module):
    """Transformer layer using DP-compatible Flash Attention"""
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = DPMultiheadAttentionWithFlashAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=False,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x_ = self.ln1(x)
        x = x + self.self_attn(x_, x_, x_)[0]
        x_ = self.ln2(x)
        x = x + self.ffn(x_)
        return x


class DPTransformerModel(nn.Module):
    """DP-compatible Transformer model for testing with Flash Attention"""
    def __init__(self, vocab_size=1000, d_model=256, nhead=4, num_layers=4, num_classes=3, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            DPCompatibleTransformerLayerWithFlashAttention(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.ln_final(x)
        x = x.mean(dim=1)  # Global average pooling
        logits = self.classifier(x)
        
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return type('Output', (), {'logits': logits, 'loss': loss})()
        return type('Output', (), {'logits': logits})()


def setup_distributed(rank, world_size):
    """Setup distributed training"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def create_dummy_data(batch_size, seq_length, vocab_size, device):
    """Create dummy data for testing"""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    labels = torch.randint(0, 3, (batch_size,), device=device)
    return {"input_ids": input_ids, "labels": labels}


def test_async_correctness_worker(rank, world_size, use_fsdp, results_dict):
    """Worker function for correctness test"""
    # Setup distributed if needed
    if use_fsdp:
        setup_distributed(rank, world_size)
    
    master_process = rank == 0
    
    if master_process:
        print("\n" + "="*80)
        print("CORRECTNESS TEST: Async vs Sync Norm Computation")
        print("="*80 + "\n")
    
    # Determine device and modes
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        if use_fsdp:
            sync_mode = "ghost_fsdp"
            async_mode = "flash_fsdp"
            if master_process:
                print(f"CUDA available with FSDP (rank {rank}/{world_size}), using FSDP modes")
        else:
            sync_mode = "ghost"
            async_mode = "flash"
            if master_process:
                print("CUDA available, using non-FSDP modes")
    else:
        device = torch.device("cpu")
        sync_mode = "ghost"
        async_mode = "flash"
        if master_process:
            print("CUDA not available, testing on CPU with non-FSDP modes")
    
    seq_length = 128
    batch_size = 4
    vocab_size = 1000
    
    # IMPORTANT: Set seed before model creation to ensure same initialization
    torch.manual_seed(42)
    
    # Test 1: Use the sync implementation
    if master_process:
        print(f"--- Testing SYNC norm computation ({sync_mode}) ---")
    
    # Create model
    model_sync = DPTransformerModel(vocab_size=vocab_size)
    
    # Wrap with FSDP if needed
    if use_fsdp:
        from opacus.utils.fsdp_utils import FSDP2Wrapper
        mp_policy = dist.fsdp.MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32
        ) if torch.cuda.is_available() else None
        model_sync = FSDP2Wrapper(model_sync, mp_policy=mp_policy)
    
    model_sync = model_sync.to(device)
    model_sync.train()
    
    optimizer_sync = torch.optim.SGD(model_sync.parameters(), lr=1e-3)
    criterion_sync = nn.CrossEntropyLoss()
    
    dummy_data = torch.randn(batch_size * 2, seq_length)
    dummy_labels = torch.randint(0, 3, (batch_size * 2,))
    dummy_dataset = TensorDataset(dummy_data, dummy_labels)
    dummy_dataloader = DataLoader(dummy_dataset, batch_size=batch_size, shuffle=False)
    
    privacy_engine_sync = PrivacyEngine()
    model_sync, optimizer_sync, criterion_sync, _ = privacy_engine_sync.make_private(
        module=model_sync,
        optimizer=optimizer_sync,
        data_loader=dummy_dataloader,
        noise_multiplier=0.0,  # No noise for correctness test
        max_grad_norm=1.0,
        grad_sample_mode=sync_mode,  # Use appropriate mode for device
        criterion=criterion_sync,
        poisson_sampling=False,
    )
    
    torch.manual_seed(42)
    batch = create_dummy_data(batch_size, seq_length, vocab_size, device)
    optimizer_sync.zero_grad()
    outputs = model_sync(input_ids=batch["input_ids"])
    loss = criterion_sync(outputs.logits, batch["labels"])
    loss.backward()
    
    sync_norms = model_sync.per_sample_gradient_norms.clone()
    if master_process:
        print(f"  Norms shape: {sync_norms.shape}")
        print(f"  Norms: {sync_norms}")
    
    # Test 2: Use the async implementation  
    if master_process:
        print(f"\n--- Testing ASYNC norm computation ({async_mode}) ---")
    
    # IMPORTANT: Reset seed to get same model initialization
    torch.manual_seed(42)
    
    # Create model
    model_async = DPTransformerModel(vocab_size=vocab_size)
    
    # Wrap with FSDP if needed (same as sync model)
    if use_fsdp:
        mp_policy = dist.fsdp.MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32
        ) if torch.cuda.is_available() else None
        model_async = FSDP2Wrapper(model_async, mp_policy=mp_policy)
    
    model_async = model_async.to(device)
    model_async.train()
    
    optimizer_async = torch.optim.SGD(model_async.parameters(), lr=1e-3)
    criterion_async = nn.CrossEntropyLoss()
    
    privacy_engine_async = PrivacyEngine()
    model_async, optimizer_async, criterion_async, _ = privacy_engine_async.make_private(
        module=model_async,
        optimizer=optimizer_async,
        data_loader=dummy_dataloader,
        noise_multiplier=0.0,  # No noise for correctness test
        max_grad_norm=1.0,
        grad_sample_mode=async_mode,  # Use appropriate mode for device
        criterion=criterion_async,
        poisson_sampling=False,
    )
    
    torch.manual_seed(42)
    batch = create_dummy_data(batch_size, seq_length, vocab_size, device)
    optimizer_async.zero_grad()
    outputs = model_async(input_ids=batch["input_ids"])
    loss = criterion_async(outputs.logits, batch["labels"])
    loss.backward()
    
    async_norms = model_async.per_sample_gradient_norms.clone()
    if master_process:
        print(f"  Norms shape: {async_norms.shape}")
        print(f"  Norms: {async_norms}")
    
    # Compare
    if master_process:
        print("\n--- Comparison ---")
        # Convert to same dtype for comparison
        sync_norms_f32 = sync_norms.float()
        async_norms_f32 = async_norms.float()
        
        diff = (sync_norms_f32 - async_norms_f32).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        rel_diff = (diff / (sync_norms_f32.abs() + 1e-10)).max().item()
        
        print(f"  Max absolute difference: {max_diff:.6e}")
        print(f"  Mean absolute difference: {mean_diff:.6e}")
        print(f"  Max relative difference: {rel_diff:.6e}")
        
        passed = torch.allclose(sync_norms_f32, async_norms_f32, rtol=1e-4, atol=1e-6)
        if passed:
            print("\n✓ CORRECTNESS TEST PASSED!")
            results_dict["passed"] = True
        else:
            print("\n✗ CORRECTNESS TEST FAILED!")
            print(f"   Tolerance exceeded: max_diff={max_diff:.6e}, rel_diff={rel_diff:.6e}")
            results_dict["passed"] = False
    
    # Cleanup
    if use_fsdp:
        cleanup_distributed()


def test_async_correctness():
    """Main entry point for correctness test"""
    # Determine if we should use FSDP
    use_fsdp = torch.cuda.is_available()  # Use FSDP if CUDA available
    
    if use_fsdp:
        world_size = torch.cuda.device_count()
        if world_size == 0:
            print("No CUDA devices available, falling back to CPU")
            use_fsdp = False
            world_size = 1
    else:
        world_size = 1
    
    print(f"Running correctness test with {world_size} GPU(s), FSDP: {use_fsdp}")
    
    # Use multiprocessing manager for sharing results
    manager = mp.Manager()
    results_dict = manager.dict()
    
    if use_fsdp and world_size > 1:
        # Multi-GPU: spawn processes
        mp.spawn(
            test_async_correctness_worker,
            args=(world_size, use_fsdp, results_dict),
            nprocs=world_size,
            join=True,
        )
    else:
        # Single GPU or CPU: run directly
        test_async_correctness_worker(0, 1, use_fsdp, results_dict)
    
    return results_dict.get("passed", False)


def test_async_vs_sync_performance_worker(rank, world_size, results_dict):
    """Worker function for performance test"""
    # Setup distributed
    setup_distributed(rank, world_size)
    
    master_process = rank == 0
    
    if master_process:
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON: Sync vs Async with FSDP")
        print("="*80 + "\n")
    
    device = torch.device(f"cuda:{rank}")
    seq_length = 256
    batch_size = 2
    vocab_size = 1000
    num_iterations = 5
    
    # Create DP-compatible model with Flash Attention + FSDP
    if master_process:
        print("Creating DP-compatible Transformer model with Flash Attention + FSDP...")
    
    from opacus.utils.fsdp_utils import FSDP2Wrapper
    
    # Test 1: Baseline - Use ghost_fsdp (sync FSDP implementation)
    if master_process:
        print("\n=== BASELINE: Synchronous FSDP (ghost_fsdp) ===")
    
    model_sync = DPTransformerModel(vocab_size=vocab_size, d_model=512, nhead=8, num_layers=6)
    mp_policy = dist.fsdp.MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32
    )
    model_sync = FSDP2Wrapper(model_sync, mp_policy=mp_policy)
    model_sync.train()
    
    optimizer_sync = torch.optim.SGD(model_sync.parameters(), lr=1e-3)
    criterion_sync = nn.CrossEntropyLoss()
    
    dummy_data = torch.randn(batch_size * world_size, seq_length)
    dummy_labels = torch.randint(0, 3, (batch_size * world_size,))
    dummy_dataset = TensorDataset(dummy_data, dummy_labels)
    dummy_dataloader = DataLoader(
        dummy_dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(dummy_dataset, num_replicas=world_size, rank=rank),
    )
    
    privacy_engine_sync = PrivacyEngine()
    model_sync, optimizer_sync, criterion_sync, _ = privacy_engine_sync.make_private(
        module=model_sync,
        optimizer=optimizer_sync,
        data_loader=dummy_dataloader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        grad_sample_mode="ghost_fsdp",  # Original sync FSDP
        criterion=criterion_sync,
        poisson_sampling=False,
    )
    
    # Warmup
    if rank == 0:
        print("Running warmup for sync...")
    for _ in range(2):
        batch = create_dummy_data(batch_size, seq_length, vocab_size, device)
        optimizer_sync.zero_grad()
        outputs = model_sync(input_ids=batch["input_ids"])
        loss = criterion_sync(outputs.logits, batch["labels"])
        loss.backward()
        optimizer_sync.step()
    
    # Benchmark sync
    if master_process:
        print("Benchmarking sync norm computation...")
    torch.cuda.reset_peak_memory_stats(device)
    
    times_sync = []
    for i in range(num_iterations):
        batch = create_dummy_data(batch_size, seq_length, vocab_size, device)
        optimizer_sync.zero_grad()
        
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        outputs = model_sync(input_ids=batch["input_ids"])
        loss = criterion_sync(outputs.logits, batch["labels"])
        loss.backward()
        optimizer_sync.step()
        
        end.record()
        torch.cuda.synchronize()
        
        iter_time = start.elapsed_time(end)
        times_sync.append(iter_time)
        
        if master_process:
            print(f"  Iteration {i+1}: {iter_time:.2f} ms")
    
    mem_sync = torch.cuda.max_memory_allocated(device) / (1024**3)
    avg_time_sync = sum(times_sync) / len(times_sync)
    
    if master_process:
        print(f"  Average time: {avg_time_sync:.2f} ms")
        print(f"  Peak memory: {mem_sync:.2f} GB")
    
    # Test 2: NEW async FSDP implementation
    if master_process:
        print("\n=== NEW: Async Stream FSDP (flash_fsdp) ===")
    
    model_async = DPTransformerModel(vocab_size=vocab_size, d_model=512, nhead=8, num_layers=6)
    model_async = FSDP2Wrapper(model_async, mp_policy=mp_policy)
    model_async.train()
    
    optimizer_async = torch.optim.SGD(model_async.parameters(), lr=1e-3)
    criterion_async = nn.CrossEntropyLoss()
    
    privacy_engine_async = PrivacyEngine()
    model_async, optimizer_async, criterion_async, _ = privacy_engine_async.make_private(
        module=model_async,
        optimizer=optimizer_async,
        data_loader=dummy_dataloader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        grad_sample_mode="flash_fsdp",  # New async FSDP
        criterion=criterion_async,
        poisson_sampling=False,
    )
    
    # Warmup
    if master_process:
        print("Running warmup for async...")
    for _ in range(2):
        batch = create_dummy_data(batch_size, seq_length, vocab_size, device)
        optimizer_async.zero_grad()
        outputs = model_async(input_ids=batch["input_ids"])
        loss = criterion_async(outputs.logits, batch["labels"])
        loss.backward()
        optimizer_async.step()
    
    # Benchmark async
    if master_process:
        print("Benchmarking async norm computation...")
    torch.cuda.reset_peak_memory_stats(device)
    
    times_async = []
    for i in range(num_iterations):
        batch = create_dummy_data(batch_size, seq_length, vocab_size, device)
        optimizer_async.zero_grad()
        
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        outputs = model_async(input_ids=batch["input_ids"])
        loss = criterion_async(outputs.logits, batch["labels"])
        loss.backward()
        optimizer_async.step()
        
        end.record()
        torch.cuda.synchronize()
        
        iter_time = start.elapsed_time(end)
        times_async.append(iter_time)
        
        if master_process:
            print(f"  Iteration {i+1}: {iter_time:.2f} ms")
    
    mem_async = torch.cuda.max_memory_allocated(device) / (1024**3)
    avg_time_async = sum(times_async) / len(times_async)
    
    if master_process:
        print(f"  Average time: {avg_time_async:.2f} ms")
        print(f"  Peak memory: {mem_async:.2f} GB")
    
    # Summary
    if master_process:
        speedup = avg_time_sync / avg_time_async
        mem_increase = ((mem_async - mem_sync) / mem_sync) * 100
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"  Sync (baseline):  {avg_time_sync:.2f} ms, {mem_sync:.2f} GB")
        print(f"  Async (new):      {avg_time_async:.2f} ms, {mem_async:.2f} GB")
        print(f"  Speedup:          {speedup:.2f}x")
        print(f"  Memory change:    {mem_increase:+.1f}%")
        print("="*80 + "\n")
        
        if speedup > 1.1:
            print("✓ PERFORMANCE TEST PASSED! (>1.1x speedup from async overlap)")
        else:
            print(f"⚠ PERFORMANCE TEST: Speedup {speedup:.2f}x < 1.1x target")
            print("  (Async stream benefits are most visible with:")
            print("   - Longer sequences (T > 512)")
            print("   - More layers (> 12 layers)")
            print("   - Larger models with significant FSDP communication overhead)")
        
        # Store results
        results_dict["speedup"] = speedup
        results_dict["mem_increase"] = mem_increase
    
    cleanup_distributed()


def test_async_vs_sync_performance():
    """Main entry point for performance test"""
    if not torch.cuda.is_available():
        print("Performance test requires CUDA")
        return
    
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print(f"Performance test works best with 2+ GPUs (found {world_size})")
        print("Running with single GPU...")
        world_size = 1
    
    print(f"Running performance test with {world_size} GPU(s)")
    
    # Use multiprocessing manager for sharing results
    manager = mp.Manager()
    results_dict = manager.dict()
    
    if world_size > 1:
        # Multi-GPU: spawn processes
        mp.spawn(
            test_async_vs_sync_performance_worker,
            args=(world_size, results_dict),
            nprocs=world_size,
            join=True,
        )
    else:
        # Single GPU: run directly
        test_async_vs_sync_performance_worker(0, 1, results_dict)


def main():
    parser = argparse.ArgumentParser(description="Test async stream norm computation")
    parser.add_argument("--mode", type=str, choices=["correctness", "performance"],
                       default="correctness",
                       help="Test mode: correctness or performance")
    args = parser.parse_args()
    
    if args.mode == "correctness":
        test_async_correctness()
    elif args.mode == "performance":
        test_async_vs_sync_performance()


if __name__ == "__main__":
    main()

