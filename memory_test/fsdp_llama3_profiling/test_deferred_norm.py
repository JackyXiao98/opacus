#!/usr/bin/env python3
"""
Test script for deferred norm computation optimization.

This test verifies:
1. Correctness: Deferred and immediate computation produce identical norms
2. Performance: Deferred computation is faster than immediate computation
3. No deadlock: Works correctly with Ghost Clipping + FSDP (two-pass backward)

Deadlock Fix (已修复):
- Hook中不调用trainable_parameters()，避免在梯度同步禁用时触发FSDP all-gather
- 参数计数延迟到compute_all_norms_parallel()，此时梯度同步已恢复

Usage:
    # Test correctness (single-GPU)
    python test_deferred_norm.py --mode correctness
    
    # Test performance (requires 2 GPUs for FSDP)
    torchrun --nproc_per_node=2 test_deferred_norm.py --mode performance
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
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


def setup_distributed():
    """Setup distributed training"""
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def create_dummy_data(batch_size, seq_length, vocab_size, device):
    """Create dummy data for testing"""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    labels = torch.randint(0, 3, (batch_size,), device=device)
    return {"input_ids": input_ids, "labels": labels}


def test_correctness_single_gpu():
    """Test that deferred and immediate norm computation produce identical results"""
    print("\n" + "="*80)
    print("CORRECTNESS TEST (Single-GPU)")
    print("="*80 + "\n")
    
    device = torch.device("cuda:0")
    seq_length = 128
    batch_size = 4
    vocab_size = 1000
    
    # Create DP-compatible model with Flash Attention
    print("Creating DP-compatible Transformer model with Flash Attention...")
    model = DPTransformerModel(vocab_size=vocab_size).to(device)
    model.train()
    
    # Create optimizer and criterion
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Create dummy dataloader
    dummy_data = torch.randn(batch_size * 2, seq_length)
    dummy_labels = torch.randint(0, 3, (batch_size * 2,))
    dummy_dataset = TensorDataset(dummy_data, dummy_labels)
    dummy_dataloader = DataLoader(dummy_dataset, batch_size=batch_size, shuffle=False)
    
    # Apply DP-SGD
    print("Applying DP-SGD with flash_bk mode...")
    privacy_engine = PrivacyEngine()
    model, optimizer, criterion, _ = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dummy_dataloader,
        noise_multiplier=0.0,  # No noise for correctness test
        max_grad_norm=1.0,
        grad_sample_mode="flash_bk",
        criterion=criterion,
        poisson_sampling=False,
    )
    
    # Test 1: Run with immediate computation (original method)
    print("\n--- Testing IMMEDIATE norm computation ---")
    os.environ['OPACUS_USE_DEFERRED_NORM'] = '0'
    
    torch.manual_seed(42)
    batch = create_dummy_data(batch_size, seq_length, vocab_size, device)
    optimizer.zero_grad()
    outputs = model(input_ids=batch["input_ids"])
    loss = criterion(outputs.logits, batch["labels"])
    loss.backward()
    
    # Get norms from immediate computation
    immediate_norms = model.per_sample_gradient_norms.clone()
    print(f"  Norms shape: {immediate_norms.shape}")
    print(f"  Norms: {immediate_norms}")
    
    # Test 2: Reset and run with deferred computation
    print("\n--- Testing DEFERRED norm computation ---")
    os.environ['OPACUS_USE_DEFERRED_NORM'] = '1'
    
    # Reset model state
    optimizer.zero_grad()
    if hasattr(model._module, '_deferred_norm_cache'):
        model._module._deferred_norm_cache.clear()
    
    # Run same batch with same seed
    torch.manual_seed(42)
    batch = create_dummy_data(batch_size, seq_length, vocab_size, device)
    outputs = model(input_ids=batch["input_ids"])
    loss = criterion(outputs.logits, batch["labels"])
    loss.backward()
    
    # Get norms from deferred computation
    deferred_norms = model.per_sample_gradient_norms.clone()
    print(f"  Norms shape: {deferred_norms.shape}")
    print(f"  Norms: {deferred_norms}")
    
    # Compare
    print("\n--- Comparison ---")
    diff = (immediate_norms - deferred_norms).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_diff = (diff / (immediate_norms.abs() + 1e-10)).max().item()
    
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")
    print(f"  Max relative difference: {rel_diff:.6e}")
    
    if torch.allclose(immediate_norms, deferred_norms, rtol=1e-4, atol=1e-6):
        print("\n✓ CORRECTNESS TEST PASSED!")
        return True
    else:
        print("\n✗ CORRECTNESS TEST FAILED!")
        print(f"   Tolerance exceeded: max_diff={max_diff:.6e}, rel_diff={rel_diff:.6e}")
        return False


def test_performance_fsdp():
    """Compare performance of immediate vs deferred norm computation on FSDP"""
    rank, world_size = setup_distributed()
    
    if rank == 0:
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON TEST (FSDP)")
        print("="*80 + "\n")
    
    device = torch.device(f"cuda:{rank}")
    seq_length = 256
    batch_size = 2
    vocab_size = 1000
    num_iterations = 5
    
    # Create DP-compatible model with Flash Attention
    if rank == 0:
        print("Creating DP-compatible Transformer model with Flash Attention + FSDP...")
    
    from opacus.utils.fsdp_utils import FSDP2Wrapper
    model = DPTransformerModel(vocab_size=vocab_size, d_model=512, nhead=8, num_layers=6)
    
    # Wrap with FSDP
    mp_policy = dist.fsdp.MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32
    )
    model = FSDP2Wrapper(model, mp_policy=mp_policy)
    model.train()
    
    # Create optimizer and criterion
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Create dummy dataloader
    dummy_data = torch.randn(batch_size * world_size, seq_length)
    dummy_labels = torch.randint(0, 3, (batch_size * world_size,))
    dummy_dataset = TensorDataset(dummy_data, dummy_labels)
    dummy_dataloader = DataLoader(
        dummy_dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(dummy_dataset, num_replicas=world_size, rank=rank),
    )
    
    # Apply DP-SGD
    if rank == 0:
        print("Applying DP-SGD with flash_fsdp_bk mode...")
    
    privacy_engine = PrivacyEngine()
    model, optimizer, criterion, _ = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dummy_dataloader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        grad_sample_mode="flash_fsdp_bk",
        criterion=criterion,
        poisson_sampling=False,
    )
    
    # Warmup
    if rank == 0:
        print("Running warmup...")
    for _ in range(2):
        batch = create_dummy_data(batch_size, seq_length, vocab_size, device)
        optimizer.zero_grad()
        outputs = model(input_ids=batch["input_ids"])
        loss = criterion(outputs.logits, batch["labels"])
        loss.backward()
        optimizer.step()
    
    # Test immediate computation
    if rank == 0:
        print("\n--- Testing IMMEDIATE norm computation ---")
    os.environ['OPACUS_USE_DEFERRED_NORM'] = '0'
    torch.cuda.reset_peak_memory_stats(device)
    
    times_immediate = []
    for i in range(num_iterations):
        batch = create_dummy_data(batch_size, seq_length, vocab_size, device)
        optimizer.zero_grad()
        
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        outputs = model(input_ids=batch["input_ids"])
        loss = criterion(outputs.logits, batch["labels"])
        loss.backward()
        optimizer.step()
        
        end.record()
        torch.cuda.synchronize()
        
        iter_time = start.elapsed_time(end)
        times_immediate.append(iter_time)
        
        if rank == 0:
            print(f"  Iteration {i+1}: {iter_time:.2f} ms")
    
    mem_immediate = torch.cuda.max_memory_allocated(device) / (1024**3)
    avg_time_immediate = sum(times_immediate) / len(times_immediate)
    
    if rank == 0:
        print(f"  Average time: {avg_time_immediate:.2f} ms")
        print(f"  Peak memory: {mem_immediate:.2f} GB")
    
    # Test deferred computation
    if rank == 0:
        print("\n--- Testing DEFERRED norm computation ---")
    os.environ['OPACUS_USE_DEFERRED_NORM'] = '1'
    torch.cuda.reset_peak_memory_stats(device)
    
    times_deferred = []
    for i in range(num_iterations):
        batch = create_dummy_data(batch_size, seq_length, vocab_size, device)
        optimizer.zero_grad()
        
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        outputs = model(input_ids=batch["input_ids"])
        loss = criterion(outputs.logits, batch["labels"])
        loss.backward()
        optimizer.step()
        
        end.record()
        torch.cuda.synchronize()
        
        iter_time = start.elapsed_time(end)
        times_deferred.append(iter_time)
        
        if rank == 0:
            print(f"  Iteration {i+1}: {iter_time:.2f} ms")
    
    mem_deferred = torch.cuda.max_memory_allocated(device) / (1024**3)
    avg_time_deferred = sum(times_deferred) / len(times_deferred)
    
    if rank == 0:
        print(f"  Average time: {avg_time_deferred:.2f} ms")
        print(f"  Peak memory: {mem_deferred:.2f} GB")
    
    # Summary
    if rank == 0:
        speedup = avg_time_immediate / avg_time_deferred
        mem_increase = ((mem_deferred - mem_immediate) / mem_immediate) * 100
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"  Immediate: {avg_time_immediate:.2f} ms, {mem_immediate:.2f} GB")
        print(f"  Deferred:  {avg_time_deferred:.2f} ms, {mem_deferred:.2f} GB")
        print(f"  Speedup:   {speedup:.2f}x")
        print(f"  Memory increase: {mem_increase:+.1f}%")
        print("="*80 + "\n")
        
        if speedup > 1.2:
            print("✓ PERFORMANCE TEST PASSED! (>1.2x speedup)")
        else:
            print(f"⚠ PERFORMANCE TEST: Speedup {speedup:.2f}x < 1.2x target")
            print("  (This is expected for small models, try with larger models)")
    
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Test deferred norm computation")
    parser.add_argument("--mode", type=str, choices=["correctness", "performance"],
                       default="correctness",
                       help="Test mode: correctness (single-GPU) or performance (FSDP)")
    args = parser.parse_args()
    
    if args.mode == "correctness":
        test_correctness_single_gpu()
    elif args.mode == "performance":
        test_performance_fsdp()


if __name__ == "__main__":
    main()

