#!/usr/bin/env python3
"""
FSDP Flash Clipping Bottleneck Profiling Script

This script profiles FSDP flash clipping to identify performance bottlenecks,
particularly focusing on:
1. Norm computation time per layer
2. All-reduce communication overhead
3. Activation memory retention
4. Sequence length scaling

Author: AI Research Engineer
Date: 2025-11-17
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import sys
import os
import time
import argparse
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from opacus.grad_sample.grad_sample_module_fast_gradient_clipping_fsdp import (
    GradSampleModuleFastGradientClippingFSDP,
)
from opacus.optimizers import DPOptimizerFastGradientClipping
from opacus.utils.fast_gradient_clipping_utils import DPLossFastGradientClipping


class SimplifiedLLMModel(nn.Module):
    """Simplified LLM-style model with only Linear layers for FSDP profiling
    
    This avoids LayerNorm/Dropout issues with functorch in FSDP mode.
    Structure mirrors typical LLM but uses only supported operations.
    """
    def __init__(self, vocab_size=50257, d_model=768, num_layers=12, 
                 dim_feedforward=3072):
        super().__init__()
        self.d_model = d_model
        
        # Token and position embeddings (GPT-style)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(16384, d_model)  # Support up to 16K context
        
        # Simplified transformer layers (Linear + ReLU only)
        # Each "transformer block" = QKV projection + FFN
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            # Attention projection (simplified - just treat as linear transform)
            self.layers.append(nn.Linear(d_model, d_model))  # QKV combined
            self.layers.append(nn.ReLU())
            
            # Feed-forward network
            self.layers.append(nn.Linear(d_model, dim_feedforward))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(dim_feedforward, d_model))
            self.layers.append(nn.ReLU())
        
        # Output projection
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights (common in LLMs)
        self.lm_head.weight = self.token_embedding.weight
    
    def forward(self, x):
        # x: [batch_size, seq_len] - token indices
        batch_size, seq_len = x.shape
        
        # Create position indices
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        hidden_states = tok_emb + pos_emb  # [B, T, D]
        
        # Transformer layers (simplified)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        return logits


def get_memory_stats(device='cuda'):
    """Get current CUDA memory statistics"""
    if not torch.cuda.is_available() or device == 'cpu':
        return {}
    
    allocated = torch.cuda.memory_allocated(device) / 2**20  # MB
    reserved = torch.cuda.memory_reserved(device) / 2**20
    max_allocated = torch.cuda.max_memory_allocated(device) / 2**20
    
    return {
        'allocated_mb': allocated,
        'reserved_mb': reserved,
        'max_allocated_mb': max_allocated
    }


def profile_forward_backward(
    model: nn.Module,
    data: torch.Tensor,
    labels: torch.Tensor,
    max_grad_norm: float = 1.0,
    use_flash_clipping: bool = True,
    use_bookkeeping: bool = False,
    device: str = 'cuda'
):
    """Profile a single forward + backward pass"""
    
    # Wrap model with FSDP GradSampleModule
    wrapped_model = GradSampleModuleFastGradientClippingFSDP(
        model,
        batch_first=True,
        max_grad_norm=max_grad_norm,
        use_ghost_clipping=True,
        use_flash_clipping=use_flash_clipping,
        loss_reduction="mean",
        enable_fastdp_bookkeeping=use_bookkeeping,
    )
    
    # Create optimizer
    base_optimizer = torch.optim.SGD(wrapped_model.parameters(), lr=0.01)
    optimizer = DPOptimizerFastGradientClipping(
        optimizer=base_optimizer,
        noise_multiplier=0.0,  # No noise for profiling
        max_grad_norm=max_grad_norm,
        expected_batch_size=data.shape[0],
        loss_reduction="mean",
    )
    
    # Create loss wrapper
    criterion = nn.CrossEntropyLoss(reduction="mean")
    dp_loss = DPLossFastGradientClipping(
        wrapped_model,
        optimizer,
        criterion,
        loss_reduction="mean",
    )
    
    # Warmup
    for _ in range(2):
        optimizer.zero_grad()
        outputs = wrapped_model(data)
        if labels.dim() == 2:
            loss_tensor = dp_loss(outputs.view(-1, outputs.shape[-1]), labels.view(-1), shape=outputs.shape[:2])
        else:
            loss_tensor = dp_loss(outputs, labels)
        loss_tensor.backward()
    
    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    # Profile forward pass
    t_forward_start = time.time()
    optimizer.zero_grad()
    outputs = wrapped_model(data)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_forward_end = time.time()
    
    mem_after_forward = get_memory_stats(device)
    
    # Profile loss computation
    t_loss_start = time.time()
    if labels.dim() == 2:
        loss_tensor = dp_loss(outputs.view(-1, outputs.shape[-1]), labels.view(-1), shape=outputs.shape[:2])
    else:
        loss_tensor = dp_loss(outputs, labels)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_loss_end = time.time()
    
    # Profile backward pass (with detailed logging enabled)
    os.environ['OPACUS_PROFILE_FSDP'] = '1'
    t_backward_start = time.time()
    loss_tensor.backward()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_backward_end = time.time()
    os.environ['OPACUS_PROFILE_FSDP'] = '0'
    
    mem_peak = get_memory_stats(device)
    
    return {
        'forward_time_ms': (t_forward_end - t_forward_start) * 1000,
        'loss_time_ms': (t_loss_end - t_loss_start) * 1000,
        'backward_time_ms': (t_backward_end - t_backward_start) * 1000,
        'total_time_ms': (t_backward_end - t_forward_start) * 1000,
        'mem_after_forward_mb': mem_after_forward.get('allocated_mb', 0),
        'mem_peak_mb': mem_peak.get('max_allocated_mb', 0),
    }


def run_profiling_suite(args):
    """Run comprehensive profiling across different configurations"""
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    if rank == 0:
        print("="*80)
        print("FSDP Flash Clipping Bottleneck Profiling")
        print("="*80)
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        print(f"Use flash clipping: {args.use_flash}")
        print(f"Use bookkeeping: {args.use_bookkeeping}")
        print()
    
    # Test configurations
    sequence_lengths = args.seq_lengths
    batch_size = args.batch_size
    
    results = []
    
    for seq_len in sequence_lengths:
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"Testing Sequence Length: {seq_len}")
            print(f"{'='*80}\n")
        
        # Create model and data
        torch.manual_seed(42 + seq_len)
        model = SimplifiedLLMModel(
            vocab_size=args.vocab_size,
            d_model=args.model_dim,
            num_layers=args.num_layers,
            dim_feedforward=args.hidden_dim
        )
        
        if device == 'cuda':
            model = model.cuda()
        
        # Create data (token indices for LLM)
        data = torch.randint(0, args.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, args.vocab_size, (batch_size, seq_len))
        
        if device == 'cuda':
            data = data.cuda()
            labels = labels.cuda()
        
        # Run profiling
        try:
            stats = profile_forward_backward(
                model=model,
                data=data,
                labels=labels,
                max_grad_norm=args.max_grad_norm,
                use_flash_clipping=args.use_flash,
                use_bookkeeping=args.use_bookkeeping,
                device=device
            )
            
            stats['seq_len'] = seq_len
            stats['batch_size'] = batch_size
            results.append(stats)
            
            if rank == 0:
                print(f"\n[Rank {rank}] Results for seq_len={seq_len}:")
                print(f"  Forward time:   {stats['forward_time_ms']:.2f} ms")
                print(f"  Loss time:      {stats['loss_time_ms']:.2f} ms")
                print(f"  Backward time:  {stats['backward_time_ms']:.2f} ms")
                print(f"  Total time:     {stats['total_time_ms']:.2f} ms")
                print(f"  Memory after forward: {stats['mem_after_forward_mb']:.2f} MB")
                print(f"  Peak memory:    {stats['mem_peak_mb']:.2f} MB")
        
        except Exception as e:
            if rank == 0:
                print(f"ERROR for seq_len={seq_len}: {e}")
                import traceback
                traceback.print_exc()
        
        # Cleanup
        del model, data, labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Summary
    if rank == 0 and results:
        print(f"\n{'='*80}")
        print("SUMMARY: Scaling with Sequence Length")
        print(f"{'='*80}")
        print(f"{'Seq Len':<10} {'Forward(ms)':<15} {'Backward(ms)':<15} {'Total(ms)':<15} {'Peak Mem(MB)':<15}")
        print("-"*80)
        for r in results:
            print(f"{r['seq_len']:<10} {r['forward_time_ms']:<15.2f} {r['backward_time_ms']:<15.2f} "
                  f"{r['total_time_ms']:<15.2f} {r['mem_peak_mb']:<15.2f}")
        
        # Analyze scaling
        if len(results) >= 2:
            print(f"\n{'='*80}")
            print("Scaling Analysis")
            print(f"{'='*80}")
            
            for i in range(1, len(results)):
                ratio = results[i]['seq_len'] / results[0]['seq_len']
                time_ratio = results[i]['backward_time_ms'] / results[0]['backward_time_ms']
                mem_ratio = results[i]['mem_peak_mb'] / results[0]['mem_peak_mb']
                
                print(f"Seq len {results[i]['seq_len']} vs {results[0]['seq_len']} "
                      f"(ratio: {ratio:.1f}x):")
                print(f"  Backward time ratio: {time_ratio:.2f}x (expected O(T): {ratio:.2f}x, O(T^2): {ratio**2:.2f}x)")
                print(f"  Memory ratio: {mem_ratio:.2f}x")


def main():
    parser = argparse.ArgumentParser(description='Profile FSDP Flash Clipping bottlenecks with LLM')
    parser.add_argument('--seq-lengths', type=int, nargs='+', default=[1024, 2048, 4096, 8192],
                        help='Sequence lengths to test')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--vocab-size', type=int, default=50257,
                        help='Vocabulary size (default: GPT-2 vocab size)')
    parser.add_argument('--model-dim', type=int, default=768,
                        help='Model dimension (d_model)')
    parser.add_argument('--hidden-dim', type=int, default=3072,
                        help='Feedforward dimension (typically 4x model_dim)')
    parser.add_argument('--num-layers', type=int, default=12,
                        help='Number of transformer layers (simplified blocks)')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')
    parser.add_argument('--use-flash', action='store_true', default=True,
                        help='Use flash clipping (triton kernels)')
    parser.add_argument('--no-flash', dest='use_flash', action='store_false',
                        help='Disable flash clipping')
    parser.add_argument('--use-bookkeeping', action='store_true',
                        help='Use bookkeeping mode (single pass)')
    parser.add_argument('--cpu', action='store_true',
                        help='Run on CPU instead of GPU')
    
    args = parser.parse_args()
    
    run_profiling_suite(args)


if __name__ == "__main__":
    main()

