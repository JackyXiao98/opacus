#!/usr/bin/env python3
"""
Run a single profiling experiment with a custom Transformer model.
Supports both single-GPU and FSDP modes.
This script is designed to be called from a shell script with different arguments.
"""

import argparse
import gc
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from opacus import PrivacyEngine
from opacus.utils.fsdp_utils import FSDP2Wrapper


# =============================================================================
# Custom Transformer Model with Flash Attention
# =============================================================================

class DPMultiheadAttentionWithFlashAttention(nn.Module):
    """
    DP-compatible MultiheadAttention using PyTorch's scaled_dot_product_attention.
    Uses F.scaled_dot_product_attention for efficient attention computation.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
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
    
    def forward(self, query, key=None, value=None, attn_mask=None, need_weights=False):
        # Self-attention: use query for all
        if key is None:
            key = query
        if value is None:
            value = query
            
        if self.batch_first:
            bsz, tgt_len, _ = query.shape
            _, src_len, _ = key.shape
        else:
            tgt_len, bsz, _ = query.shape
            src_len, bsz, _ = key.shape
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape: (B, L, D) -> (B, L, H, D_head) -> (B, H, L, D_head)
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Use PyTorch's scaled_dot_product_attention (Flash Attention when available)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )
        
        # Reshape back: (B, H, L, D_head) -> (B, L, H, D_head) -> (B, L, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        return output, None


class TransformerLayerWithFlashAttention(nn.Module):
    """Transformer layer using Flash Attention."""
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
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
        x = x + self.self_attn(x_)[0]
        x_ = self.ln2(x)
        x = x + self.ffn(x_)
        return x


class TransformerModelWithFlashAttention(nn.Module):
    """
    Transformer model using Flash Attention for benchmarking.
    Similar to a small LLM with embedding, transformer layers, and lm_head.
    """
    def __init__(
        self, 
        vocab_size: int, 
        hidden_dim: int, 
        num_layers: int, 
        num_heads: int, 
        seq_len: int
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(seq_len, hidden_dim)
        self.seq_len = seq_len
        self.layers = nn.ModuleList([
            TransformerLayerWithFlashAttention(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        x = self.embedding(input_ids) + self.pos_embedding(position_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(x)

        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Helper Functions
# =============================================================================

def aggressive_cleanup():
    """Aggressive memory cleanup"""
    for _ in range(3):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    gc.collect(generation=2)


def setup(rank, world_size):
    """Setup distributed training"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"  # Different port from single_experiment.py
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """Cleanup distributed training"""
    dist.destroy_process_group()


def create_model(config, master_process=True):
    """Create custom transformer model"""
    if master_process:
        print(f"Creating Transformer model: vocab_size={config['vocab_size']}, "
              f"hidden_dim={config['hidden_dim']}, num_layers={config['num_layers']}, "
              f"num_heads={config['num_heads']}, seq_len={config['seq_length']}")
    
    model = TransformerModelWithFlashAttention(
        vocab_size=config["vocab_size"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        seq_len=config["seq_length"],
    )
    
    if master_process:
        print(f"Model parameters: {model.count_parameters():,}")
    
    return model


def generate_batch(config, device):
    """Generate synthetic batch for transformer"""
    batch_size = config["batch_size"]
    seq_length = config["seq_length"]
    vocab_size = config["vocab_size"]
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    return {"input_ids": input_ids, "labels": labels}


# =============================================================================
# Experiment Runner
# =============================================================================

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
    """Run experiment on a single GPU worker"""
    torch.cuda.set_device(rank)
    
    # Determine if this is an FSDP mode
    is_fsdp_mode = mode in [
        "ghost_fsdp", "flash_fsdp", "flash_fsdp_bk", "ghost_fsdp_bk", 
        "flash_fsdp_fuse", "flash_fsdp_fuse_bk", "no_dp"
    ]
    
    # Only setup distributed for FSDP modes
    if is_fsdp_mode:
        setup(rank, world_size)
    
    master_process = rank == 0
    torch.manual_seed(1337 + rank)
    
    seq_length = config["seq_length"]
    batch_size = config["batch_size"]
    
    if master_process:
        print(f"\n{'='*80}")
        print(f"EXPERIMENT: mode={mode}, seq_length={seq_length}, batch_size={batch_size}")
        print(f"{'='*80}\n")
    
    device = torch.device(f"cuda:{rank}")
    
    # Create model
    model = create_model(config, master_process)
    
    # Determine if this is a fuse mode (needs special handling)
    is_fuse_mode = mode in ["flash_fsdp_fuse", "flash_fsdp_fuse_bk", "flash_fuse", "flash_fuse_bk"]
    
    # Determine if we need to pass shape parameter to criterion
    # All DP modes (except hooks/grad_materialize) wrap criterion with DPLossFastGradientClipping
    # which needs shape=(B, T, V) to reduce per-token losses [B*T] to per-sample [B]
    # Note: For transformers, even fuse modes need shape param because CrossEntropyLoss
    # requires flattened input (unlike MSELoss which can handle 3D directly)
    needs_shape_param = mode in [
        "ghost", "ghost_bk", "ghost_fsdp", "ghost_fsdp_bk",
        "flash", "flash_bk", "flash_fsdp", "flash_fsdp_bk",
        "flash_fuse", "flash_fuse_bk", "flash_fsdp_fuse", "flash_fsdp_fuse_bk"
    ]
    
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
    
    # Create criterion (CrossEntropy for language modeling)
    criterion = nn.CrossEntropyLoss()
    
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
            dummy_labels = torch.randint(0, config["vocab_size"], (batch_size * world_size,))
            dummy_dataset = TensorDataset(dummy_data, dummy_labels)
            dummy_dataloader = DataLoader(
                dummy_dataset,
                batch_size=batch_size,
                sampler=DistributedSampler(dummy_dataset, num_replicas=world_size, rank=rank),
            )
        else:
            dummy_data = torch.randn(batch_size, seq_length)
            dummy_labels = torch.randint(0, config["vocab_size"], (batch_size,))
            dummy_dataset = TensorDataset(dummy_data, dummy_labels)
            dummy_dataloader = DataLoader(
                dummy_dataset,
                batch_size=batch_size,
                shuffle=False,
            )
        
        # Map grad_materialize to "hooks" mode (standard Opacus)
        grad_sample_mode = "hooks" if mode == "grad_materialize" else mode
        
        # For hooks mode, don't pass criterion (it doesn't need wrapping)
        if grad_sample_mode == "hooks":
            model, optimizer, *_ = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=dummy_dataloader,
                noise_multiplier=sigma,
                max_grad_norm=max_grad_norm,
                grad_sample_mode=grad_sample_mode,
                poisson_sampling=False,
            )
        else:
            model, optimizer, criterion, *_ = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=dummy_dataloader,
                noise_multiplier=sigma,
                max_grad_norm=max_grad_norm,
                grad_sample_mode=grad_sample_mode,
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
        batch = generate_batch(config, device)
        
        optimizer.zero_grad()
        
        logits = model(input_ids=batch["input_ids"])
        B, T, V = logits.shape
        
        if needs_shape_param:
            # Ghost/flash/fuse modes: pass shape=(B, T, V) so DPLossFastGradientClipping
            # can reduce per-token losses [B*T] to per-sample losses [B]
            # Note: CrossEntropyLoss requires flattened input unlike MSELoss
            loss = criterion(logits.view(-1, V), batch["labels"].view(-1), shape=(B, T, V))
        else:
            # Standard modes (no_dp, no_dp_single, grad_materialize)
            loss = criterion(logits.view(-1, V), batch["labels"].view(-1))
        
        loss.backward()
        optimizer.step()
        
        if is_fsdp_mode:
            torch.cuda.synchronize()
            if dist.is_initialized():
                dist.barrier()
        
        del batch, loss, logits
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
        batch = generate_batch(config, device)
        
        # Time the iteration
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        
        optimizer.zero_grad()
        
        logits = model(input_ids=batch["input_ids"])
        B, T, V = logits.shape
        
        if needs_shape_param:
            # Ghost/flash/fuse modes: pass shape=(B, T, V) so DPLossFastGradientClipping
            # can reduce per-token losses [B*T] to per-sample losses [B]
            # Note: CrossEntropyLoss requires flattened input unlike MSELoss
            loss = criterion(logits.view(-1, V), batch["labels"].view(-1), shape=(B, T, V))
        else:
            # Standard modes (no_dp, no_dp_single, grad_materialize)
            loss = criterion(logits.view(-1, V), batch["labels"].view(-1))
        
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
        
        del batch, loss, logits
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


def run_experiment(config):
    """Run the experiment with the given configuration"""
    mode = config["mode"]
    
    # Determine if this is an FSDP mode (multi-GPU)
    is_fsdp_mode = mode in [
        "ghost_fsdp", "flash_fsdp", "flash_fsdp_bk", "ghost_fsdp_bk", 
        "flash_fsdp_fuse", "flash_fsdp_fuse_bk", "no_dp"
    ]
    
    if is_fsdp_mode:
        world_size = torch.cuda.device_count()
        if world_size == 0:
            raise RuntimeError("No CUDA devices available")
    else:
        world_size = 1
        if torch.cuda.device_count() == 0:
            raise RuntimeError("No CUDA devices available")
    
    seq_length = config["seq_length"]
    
    print(f"\n{'#'*80}")
    print(f"Configuration:")
    print(f"  Mode: {mode}")
    print(f"  Vocab Size: {config['vocab_size']}")
    print(f"  Hidden Dim: {config['hidden_dim']}")
    print(f"  Num Layers: {config['num_layers']}")
    print(f"  Num Heads: {config['num_heads']}")
    print(f"  Sequence Length: {seq_length}")
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
    
    # Extract results
    results = {
        "mode": mode,
        "seq_length": seq_length,
        "batch_size": config["batch_size"],
        "total_batch_size": config["batch_size"] * world_size,
        "num_gpus": world_size,
        "vocab_size": config["vocab_size"],
        "hidden_dim": config["hidden_dim"],
        "num_layers": config["num_layers"],
        "num_heads": config["num_heads"],
        "peak_memory_mb": results_dict.get("peak_memory_mb", 0),
        "peak_memory_gb": results_dict.get("peak_memory_gb", 0),
        "avg_time_ms": results_dict.get("avg_time_ms", 0),
        "config": config,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run single transformer profiling experiment")
    
    # Mode arguments
    parser.add_argument("--mode", type=str, required=True,
                       choices=[
                           # FSDP modes
                           "no_dp", "ghost_fsdp", "flash_fsdp", "flash_fsdp_bk", 
                           "ghost_fsdp_bk", "flash_fsdp_fuse", "flash_fsdp_fuse_bk",
                           # Single-GPU modes
                           "no_dp_single", "grad_materialize", "ghost", "ghost_bk", 
                           "flash", "flash_bk", "flash_fuse", "flash_fuse_bk"
                       ],
                       help="Training mode")
    
    # Model architecture arguments
    parser.add_argument("--vocab-size", type=int, default=32000,
                       help="Vocabulary size")
    parser.add_argument("--hidden-dim", type=int, default=512,
                       help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=4,
                       help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=8,
                       help="Number of attention heads")
    parser.add_argument("--seq-length", type=int, default=1024,
                       help="Sequence length")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size per GPU")
    parser.add_argument("--num-iter", type=int, default=3,
                       help="Number of profiling iterations")
    parser.add_argument("--warmup-iter", type=int, default=1,
                       help="Number of warmup iterations")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--sigma", type=float, default=1.0,
                       help="Noise multiplier for DP")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                       help="Max gradient norm for DP")
    
    # Output arguments
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSON file path")
    
    args = parser.parse_args()
    
    # Build config
    config = {
        "mode": args.mode,
        "vocab_size": args.vocab_size,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "seq_length": args.seq_length,
        "batch_size": args.batch_size,
        "num_iter": args.num_iter,
        "warmup_iter": args.warmup_iter,
        "learning_rate": args.learning_rate,
        "sigma": args.sigma,
        "max_grad_norm": args.max_grad_norm,
    }
    
    # Run experiment
    aggressive_cleanup()
    results = run_experiment(config)
    aggressive_cleanup()
    
    # Save results
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
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

