#!/usr/bin/env python3
"""
Test file for flash_fuse and flash_fuse_bk modes.

This script tests:
1. Consistency: Verify flash_fuse and flash_fuse_bk produce identical gradients 
   to ghost, ghost_bk, flash, flash_bk using simple linear layers
2. Speed: Benchmark all 6 modes with sequence_length=16384
"""

import time
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from opacus.grad_sample.utils import wrap_model
from opacus.utils.fast_gradient_clipping_utils import DPLossFastGradientClipping


# =============================================================================
# Test Model
# =============================================================================

class SimpleLinearModel(nn.Module):
    """Simple 2-layer Linear model for testing."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class PerSampleMSELoss(nn.Module):
    """MSE Loss that reduces to per-sample loss [batch_size]."""
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute MSE per element: [B, D] or [B, T, D]
        mse = (input - target) ** 2
        # Reduce all dims except batch to get per-sample loss
        if mse.dim() == 2:
            # [B, D] -> [B]
            per_sample = mse.mean(dim=1)
        elif mse.dim() == 3:
            # [B, T, D] -> [B]
            per_sample = mse.mean(dim=(1, 2))
        else:
            # Sum all except first dim
            per_sample = mse.view(mse.shape[0], -1).mean(dim=1)
        return per_sample


class PerSampleCrossEntropyLoss(nn.Module):
    """CrossEntropy Loss that reduces to per-sample loss [batch_size]."""
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits: [B, T, V] or [B, V], labels: [B, T] or [B]
        if logits.dim() == 3:
            # [B, T, V] -> [B*T, V]
            B, T, V = logits.shape
            logits_flat = logits.view(-1, V)
            labels_flat = labels.view(-1)
            # Compute per-token loss
            loss_flat = nn.functional.cross_entropy(logits_flat, labels_flat, reduction='none')
            # Reshape back to [B, T] and mean over T
            per_sample = loss_flat.view(B, T).mean(dim=1)
        else:
            # [B, V] -> [B]
            per_sample = nn.functional.cross_entropy(logits, labels, reduction='none')
        return per_sample


# =============================================================================
# Transformer Model with Flash Attention
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
        attn_output = nn.functional.scaled_dot_product_attention(
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

def clone_model(model: nn.Module) -> nn.Module:
    """Create a deep copy of a model with the same weights."""
    import copy
    return copy.deepcopy(model)


def get_param_grads(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Get gradients from model parameters."""
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.clone()
    return grads


def compare_grads(grads1: Dict[str, torch.Tensor], grads2: Dict[str, torch.Tensor], 
                  mode1: str, mode2: str, rtol: float = 1e-4, atol: float = 1e-6) -> bool:
    """Compare gradients from two modes."""
    if set(grads1.keys()) != set(grads2.keys()):
        print(f"  [FAIL] {mode1} vs {mode2}: Different parameter sets")
        return False
    
    all_match = True
    for name in grads1.keys():
        g1, g2 = grads1[name], grads2[name]
        if not torch.allclose(g1, g2, rtol=rtol, atol=atol):
            max_diff = (g1 - g2).abs().max().item()
            print(f"  [FAIL] {mode1} vs {mode2}: {name} max_diff={max_diff:.6e}")
            all_match = False
    
    if all_match:
        print(f"  [PASS] {mode1} vs {mode2}: All gradients match")
    
    return all_match


def run_single_iteration(
    model: nn.Module,
    mode: str,
    x: torch.Tensor,
    target: torch.Tensor,
    max_grad_norm: float = 1.0,
    loss_reduction: str = "mean",
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Run a single training iteration with specified mode.
    
    Returns:
        Tuple of (param_grads, per_sample_norms)
    """
    # Clone model to avoid state pollution
    model = clone_model(model)
    
    # Wrap model with specified mode
    wrapped = wrap_model(
        model,
        grad_sample_mode=mode,
        batch_first=True,
        loss_reduction=loss_reduction,
        max_grad_norm=max_grad_norm,
    )
    
    criterion = PerSampleMSELoss(reduction=loss_reduction)
    
    # Create optimizer using get_optimizer_class to select the correct optimizer based on mode
    from opacus.optimizers import get_optimizer_class
    optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.01)
    optim_class = get_optimizer_class(clipping="flat", distributed=False, grad_sample_mode=mode)
    dp_optimizer = optim_class(
        optimizer=optimizer,
        noise_multiplier=0.0,  # No noise for testing
        max_grad_norm=max_grad_norm,
        expected_batch_size=x.shape[0],
        loss_reduction=loss_reduction,
    )
    
    # Create DPLoss wrapper for two-pass clipping
    dp_loss = DPLossFastGradientClipping(
        wrapped, dp_optimizer, criterion, loss_reduction
    )
    
    # Forward + backward (two-pass or bookkeeping depending on mode)
    wrapped.zero_grad()
    # First do forward pass through model
    output = wrapped(x)
    # Then compute loss with criterion (dp_loss wraps the criterion)
    loss = dp_loss(output, target)
    loss.backward()  # This actually runs the gradient clipping
    
    # Get per-sample norms (computed during backward)
    per_sample_norms = wrapped.per_sample_gradient_norms.clone()
    
    # Get gradients
    grads = get_param_grads(wrapped._module)
    
    return grads, per_sample_norms


# =============================================================================
# Consistency Test
# =============================================================================

def test_consistency():
    """Test that all modes produce consistent gradients."""
    print("=" * 70)
    print("CONSISTENCY TEST")
    print("=" * 70)
    
    # Test configuration - use 2D tensors (no sequence dim) for simplicity
    batch_size = 8
    input_dim = 64
    hidden_dim = 128
    output_dim = 64  # Same as input_dim to avoid dimension mismatch
    max_grad_norm = 1.0
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Create model and data (2D tensors - no sequence dimension)
    base_model = SimpleLinearModel(input_dim, hidden_dim, output_dim)
    x = torch.randn(batch_size, input_dim)  # [B, D_in]
    # Target must match model output shape
    with torch.no_grad():
        sample_out = base_model(x)
    target = torch.randn_like(sample_out)  # [B, D_out]
    
    # Modes to test (non-bookkeeping modes first, then bookkeeping modes)
    non_bk_modes = ["ghost", "flash", "flash_fuse"]
    bk_modes = ["ghost_bk", "flash_bk", "flash_fuse_bk"]
    
    print(f"\nConfig: batch_size={batch_size}, "
          f"input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}")
    print(f"max_grad_norm={max_grad_norm}")
    print()
    
    # Run all modes
    results = {}
    for mode in non_bk_modes + bk_modes:
        print(f"Running mode: {mode}...")
        grads, norms = run_single_iteration(
            base_model, mode, x, target, max_grad_norm
        )
        results[mode] = {"grads": grads, "norms": norms}
    
    print()
    print("-" * 70)
    print("GRADIENT COMPARISON")
    print("-" * 70)
    
    all_pass = True
    
    # Compare non-bookkeeping modes
    print("\nNon-bookkeeping modes (two-pass):")
    for i, mode1 in enumerate(non_bk_modes):
        for mode2 in non_bk_modes[i+1:]:
            if not compare_grads(results[mode1]["grads"], results[mode2]["grads"], mode1, mode2):
                all_pass = False
    
    # Compare bookkeeping modes
    print("\nBookkeeping modes (single-pass):")
    for i, mode1 in enumerate(bk_modes):
        for mode2 in bk_modes[i+1:]:
            if not compare_grads(results[mode1]["grads"], results[mode2]["grads"], mode1, mode2):
                all_pass = False
    
    # Compare non-bk vs bk (should match)
    print("\nCross-comparison (non-bk vs bk):")
    for non_bk, bk in zip(non_bk_modes, bk_modes):
        if not compare_grads(results[non_bk]["grads"], results[bk]["grads"], non_bk, bk):
            all_pass = False
    
    print()
    print("-" * 70)
    print("PER-SAMPLE NORM COMPARISON")
    print("-" * 70)
    
    # Compare per-sample norms
    print("\nPer-sample gradient norms:")
    for mode in non_bk_modes + bk_modes:
        norms = results[mode]["norms"]
        print(f"  {mode:20s}: mean={norms.mean():.6f}, std={norms.std():.6f}, "
              f"min={norms.min():.6f}, max={norms.max():.6f}")
    
    # Compare norms between modes
    print("\nNorm consistency check:")
    ref_norms = results["ghost"]["norms"]
    for mode in non_bk_modes + bk_modes:
        if mode == "ghost":
            continue
        mode_norms = results[mode]["norms"]
        if torch.allclose(ref_norms, mode_norms, rtol=1e-4, atol=1e-6):
            print(f"  [PASS] ghost vs {mode}: Norms match")
        else:
            max_diff = (ref_norms - mode_norms).abs().max().item()
            print(f"  [FAIL] ghost vs {mode}: max_diff={max_diff:.6e}")
            all_pass = False
    
    print()
    if all_pass:
        print("=" * 70)
        print("CONSISTENCY TEST: ALL PASSED")
        print("=" * 70)
    else:
        print("=" * 70)
        print("CONSISTENCY TEST: SOME FAILURES")
        print("=" * 70)
    
    return all_pass


# =============================================================================
# Speed Benchmark
# =============================================================================

def benchmark_mode(
    model: nn.Module,
    mode: str,
    x: torch.Tensor,
    target: torch.Tensor,
    max_grad_norm: float = 1.0,
    num_warmup: int = 3,
    num_iterations: int = 10,
) -> float:
    """
    Benchmark a single mode.
    
    Returns:
        Average time per iteration in seconds
    """
    # Wrap model with specified mode
    model_copy = clone_model(model)
    wrapped = wrap_model(
        model_copy,
        grad_sample_mode=mode,
        batch_first=True,
        loss_reduction="mean",
        max_grad_norm=max_grad_norm,
    )
    
    criterion = PerSampleMSELoss(reduction="mean")
    
    # Create optimizer using get_optimizer_class to select the correct optimizer based on mode
    from opacus.optimizers import get_optimizer_class
    optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.01)
    optim_class = get_optimizer_class(clipping="flat", distributed=False, grad_sample_mode=mode)
    dp_optimizer = optim_class(
        optimizer=optimizer,
        noise_multiplier=0.0,
        max_grad_norm=max_grad_norm,
        expected_batch_size=x.shape[0],
        loss_reduction="mean",
    )
    
    # Create DPLoss wrapper
    dp_loss = DPLossFastGradientClipping(
        wrapped, dp_optimizer, criterion, "mean"
    )
    
    # Warmup
    for _ in range(num_warmup):
        wrapped.zero_grad()
        loss = dp_loss(x, target)
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        wrapped.zero_grad()
        
        start = time.perf_counter()
        loss = dp_loss(x, target)
        end = time.perf_counter()
        
        times.append(end - start)
    
    return sum(times) / len(times)


def test_speed():
    """Benchmark speed of all modes with long sequence."""
    print()
    print("=" * 70)
    print("SPEED BENCHMARK")
    print("=" * 70)
    
    # Test configuration - long sequence
    batch_size = 4
    seq_len = 16384  # Long sequence as requested
    input_dim = 512
    hidden_dim = 512
    output_dim = 512
    max_grad_norm = 1.0
    
    print(f"\nConfig: batch_size={batch_size}, seq_len={seq_len}, "
          f"input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}")
    print()
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Create model and data
    base_model = SimpleLinearModel(input_dim, hidden_dim, output_dim)
    x = torch.randn(batch_size, seq_len, input_dim)
    # Target must match model output shape
    with torch.no_grad():
        sample_out = base_model(x)
    target = torch.randn_like(sample_out)
    
    # All modes to benchmark
    modes = ["ghost", "ghost_bk", "flash", "flash_bk", "flash_fuse", "flash_fuse_bk"]
    
    print("Running benchmarks...")
    print("-" * 70)
    
    results = {}
    for mode in modes:
        print(f"  Benchmarking {mode}...", end=" ", flush=True)
        try:
            avg_time = benchmark_mode(base_model, mode, x, target, max_grad_norm)
            results[mode] = avg_time
            print(f"{avg_time*1000:.2f} ms")
        except Exception as e:
            print(f"FAILED: {e}")
            results[mode] = float('inf')
    
    print()
    print("-" * 70)
    print("RESULTS (sorted by speed, fastest first)")
    print("-" * 70)
    
    # Sort by time
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    
    print()
    print(f"{'Rank':<6} {'Mode':<20} {'Time (ms)':<15} {'Relative':<15}")
    print("-" * 56)
    
    baseline = sorted_results[0][1]
    for i, (mode, time_s) in enumerate(sorted_results, 1):
        time_ms = time_s * 1000
        relative = time_s / baseline if baseline > 0 else float('inf')
        print(f"{i:<6} {mode:<20} {time_ms:<15.2f} {relative:<15.2f}x")
    
    print()
    print("=" * 70)
    print("SPEED BENCHMARK COMPLETE")
    print("=" * 70)


def benchmark_transformer_mode(
    model: nn.Module,
    mode: str,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    max_grad_norm: float = 1.0,
    num_warmup: int = 2,
    num_iterations: int = 5,
) -> float:
    """
    Benchmark a single mode with transformer model.
    
    Returns:
        Average time per iteration in seconds
    """
    # Wrap model with specified mode
    model_copy = clone_model(model)
    wrapped = wrap_model(
        model_copy,
        grad_sample_mode=mode,
        batch_first=True,
        loss_reduction="mean",
        max_grad_norm=max_grad_norm,
    )
    
    criterion = PerSampleCrossEntropyLoss(reduction="mean")
    
    # Create optimizer using get_optimizer_class
    from opacus.optimizers import get_optimizer_class
    optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.01)
    optim_class = get_optimizer_class(clipping="flat", distributed=False, grad_sample_mode=mode)
    dp_optimizer = optim_class(
        optimizer=optimizer,
        noise_multiplier=0.0,
        max_grad_norm=max_grad_norm,
        expected_batch_size=input_ids.shape[0],
        loss_reduction="mean",
    )
    
    # Create DPLoss wrapper
    dp_loss = DPLossFastGradientClipping(
        wrapped, dp_optimizer, criterion, "mean"
    )
    
    # Warmup
    for _ in range(num_warmup):
        wrapped.zero_grad()
        logits = wrapped(input_ids)
        loss = dp_loss(logits, labels)
        loss.backward()
        dp_optimizer.step()
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        wrapped.zero_grad()
        
        start = time.perf_counter()
        logits = wrapped(input_ids)
        loss = dp_loss(logits, labels)
        loss.backward()
        dp_optimizer.step()
        end = time.perf_counter()
        
        times.append(end - start)
    
    return sum(times) / len(times)


def test_transformer_speed():
    """Benchmark speed with Transformer model (Flash Attention)."""
    print()
    print("=" * 70)
    print("TRANSFORMER SPEED BENCHMARK (Flash Attention)")
    print("=" * 70)
    
    # Test configuration
    vocab_size = 32000
    hidden_dim = 512
    num_layers = 1
    num_heads = 1
    seq_len = 4096
    batch_size = 1
    max_grad_norm = 1.0
    
    print(f"\nConfig:")
    print(f"  vocab_size={vocab_size}, hidden_dim={hidden_dim}")
    print(f"  num_layers={num_layers}, num_heads={num_heads}")
    print(f"  seq_len={seq_len}, batch_size={batch_size}")
    print()
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Create model
    base_model = TransformerModelWithFlashAttention(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        seq_len=seq_len,
    )
    print(f"Model parameters: {base_model.count_parameters():,}")
    
    # Create data
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # All modes to benchmark
    modes = ["flash_fuse_bk", "flash_bk"]
    # , "ghost_bk", "flash_fuse", "flash", "ghost"

    print("\nRunning benchmarks...")
    print("-" * 70)
    
    results = {}
    for mode in modes:
        print(f"  Benchmarking {mode}...", end=" ", flush=True)
        try:
            avg_time = benchmark_transformer_mode(
                base_model, mode, input_ids, labels, max_grad_norm,
                num_warmup=1, num_iterations=2
            )
            results[mode] = avg_time
            print(f"{avg_time*1000:.2f} ms")
        except Exception as e:
            print(f"FAILED: {e}")
            results[mode] = float('inf')
    
    print()
    print("-" * 70)
    print("RESULTS (sorted by speed, fastest first)")
    print("-" * 70)
    
    # Sort by time
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    
    print()
    print(f"{'Rank':<6} {'Mode':<20} {'Time (ms)':<15} {'Relative':<15}")
    print("-" * 56)
    
    baseline = sorted_results[0][1]
    for i, (mode, time_s) in enumerate(sorted_results, 1):
        time_ms = time_s * 1000
        relative = time_s / baseline if baseline > 0 else float('inf')
        print(f"{i:<6} {mode:<20} {time_ms:<15.2f} {relative:<15.2f}x")
    
    print()
    print("=" * 70)
    print("TRANSFORMER SPEED BENCHMARK COMPLETE")
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("FLASH_FUSE MODE TESTING")
    print("=" * 70 + "\n")
    
    # Run consistency test
    consistency_passed = test_consistency()
    
    # Run speed benchmark with simple linear model
    # test_speed()
    
    # Run speed benchmark with transformer model
    test_transformer_speed()
    
    print("\n" + "=" * 70)
    print("TESTING COMPLETE")
    print("=" * 70)
    
    if consistency_passed:
        print("\nAll consistency tests PASSED.")
    else:
        print("\nSome consistency tests FAILED.")

