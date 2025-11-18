#!/usr/bin/env python3
"""
Test DiT model with Opacus DP-SGD: verify that multi-input conditional models work correctly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))

from opacus import GradSampleModule
from opacus.validators import ModuleValidator


class SimplifiedTimestepEmbedder(nn.Module):
    """Simplified timestep embedder for testing"""
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, t):
        """
        Args:
            t: timestep tensor (B,) or (B, 1)
        Returns:
            timestep embeddings (B, hidden_dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1).float()
        elif t.dim() == 2 and t.shape[1] == 1:
            t = t.float()
        else:
            raise ValueError(f"Unexpected timestep shape: {t.shape}")
        return self.mlp(t)


class SimplifiedLabelEmbedder(nn.Module):
    """Simplified label embedder for testing"""
    def __init__(self, num_classes=10, hidden_dim=128):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes, hidden_dim)
    
    def forward(self, labels):
        """
        Args:
            labels: class labels (B,)
        Returns:
            label embeddings (B, hidden_dim)
        """
        return self.embedding_table(labels)


class SimplifiedDiTBlock(nn.Module):
    """Simplified DiT block for testing conditional generation"""
    def __init__(self, hidden_dim=128, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        # Conditioning projection
        self.cond_proj = nn.Linear(hidden_dim, hidden_dim * 2)
    
    def forward(self, x, c):
        """
        Args:
            x: input tensor (B, N, hidden_dim)
            c: conditioning vector (B, hidden_dim)
        Returns:
            output tensor (B, N, hidden_dim)
        """
        # Project conditioning to get modulation parameters
        cond_params = self.cond_proj(c)  # (B, hidden_dim * 2)
        scale, shift = cond_params.chunk(2, dim=-1)  # Each (B, hidden_dim)
        
        # Self-attention with conditioning
        x_norm = self.norm1(x)
        # Apply conditioning
        x_modulated = x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        attn_output, _ = self.attn(x_modulated, x_modulated, x_modulated)
        x = x + attn_output
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x


class SimplifiedDiTModel(nn.Module):
    """
    Simplified DiT model with multi-input forward signature.
    Tests the key challenge: forward(x, t, y) with multiple conditional inputs.
    """
    def __init__(
        self,
        seq_len=16,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        num_classes=10,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, hidden_dim))
        
        # Timestep and label embedders
        self.timestep_embedder = SimplifiedTimestepEmbedder(hidden_dim)
        self.label_embedder = SimplifiedLabelEmbedder(num_classes, hidden_dim)
        
        # DiT blocks
        self.blocks = nn.ModuleList([
            SimplifiedDiTBlock(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x, t, y):
        """
        Multi-input forward signature - the key challenge for Opacus!
        
        Args:
            x: input data (B, seq_len, hidden_dim)
            t: timesteps (B,)
            y: class labels (B,)
        Returns:
            output (B, seq_len, hidden_dim)
        """
        B, L, D = x.shape
        assert L == self.seq_len and D == self.hidden_dim
        
        # Project input and add positional embedding
        x = self.input_proj(x) + self.pos_embed
        
        # Compute conditioning vector from timestep and label
        t_emb = self.timestep_embedder(t)  # (B, hidden_dim)
        y_emb = self.label_embedder(y)      # (B, hidden_dim)
        c = t_emb + y_emb                   # (B, hidden_dim)
        
        # Apply DiT blocks with conditioning
        for block in self.blocks:
            x = block(x, c)
        
        # Output projection
        x = self.output_proj(x)
        
        return x


def test_dit_forward_pass():
    """Test that DiT model forward pass works with multiple inputs"""
    print("\n" + "="*60)
    print("Test 1: DiT Forward Pass with Multiple Inputs")
    print("="*60)
    
    device = "cpu"
    batch_size = 4
    seq_len = 16
    hidden_dim = 128
    num_classes = 10
    
    # Create model
    model = SimplifiedDiTModel(
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=4,
        num_classes=num_classes,
    ).to(device)
    model = ModuleValidator.fix(model)
    
    # Create sample data
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)
    
    # Forward pass WITHOUT GradSampleModule (baseline)
    model.train()
    output = model(x, t, y)
    assert output.shape == (batch_size, seq_len, hidden_dim), \
        f"Expected shape {(batch_size, seq_len, hidden_dim)}, got {output.shape}"
    
    print("✓ DiT forward pass test passed!")
    print(f"  - Input shapes: x={x.shape}, t={t.shape}, y={y.shape}")
    print(f"  - Output shape: {output.shape}")
    return True


def test_dit_with_grad_sample_module():
    """Test that DiT works with GradSampleModule (multi-input support)"""
    print("\n" + "="*60)
    print("Test 2: DiT with GradSampleModule (Multi-Input)")
    print("="*60)
    
    device = "cpu"
    batch_size = 4
    seq_len = 16
    hidden_dim = 128
    num_classes = 10
    
    # Create model
    model = SimplifiedDiTModel(
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=4,
        num_classes=num_classes,
    ).to(device)
    model = ModuleValidator.fix(model)
    
    # Wrap with GradSampleModule
    gs_model = GradSampleModule(model, batch_first=True, loss_reduction="mean")
    
    # Create sample data
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)
    target = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    # Forward pass
    gs_model.train()
    output = gs_model(x, t, y)
    assert output.shape == (batch_size, seq_len, hidden_dim), \
        f"Expected shape {(batch_size, seq_len, hidden_dim)}, got {output.shape}"
    
    # Backward pass
    loss = F.mse_loss(output, target)
    loss.backward()
    
    # Check grad_sample exists
    grad_sample_count = 0
    for name, param in gs_model.named_parameters():
        if param.requires_grad:
            assert hasattr(param, 'grad_sample'), f"Parameter {name} missing grad_sample"
            assert param.grad_sample is not None, f"Parameter {name} has None grad_sample"
            assert param.grad_sample.shape[0] == batch_size, \
                f"Parameter {name} grad_sample batch dimension is {param.grad_sample.shape[0]}, expected {batch_size}"
            grad_sample_count += 1
    
    print("✓ DiT with GradSampleModule test passed!")
    print(f"  - Forward/backward completed successfully")
    print(f"  - {grad_sample_count} parameters have correct grad_sample shape")
    return True


def test_dit_conditioning_effect():
    """Test that conditioning (t, y) actually affects the output"""
    print("\n" + "="*60)
    print("Test 3: DiT Conditioning Effect")
    print("="*60)
    
    device = "cpu"
    batch_size = 2
    seq_len = 8
    hidden_dim = 64
    num_classes = 5
    
    # Create model
    torch.manual_seed(42)
    model = SimplifiedDiTModel(
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_layers=1,
        num_heads=2,
        num_classes=num_classes,
    ).to(device)
    model.eval()
    
    # Same input x, different conditioning
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    # Condition 1: t=0, y=0
    t1 = torch.zeros(batch_size, dtype=torch.long, device=device)
    y1 = torch.zeros(batch_size, dtype=torch.long, device=device)
    output1 = model(x, t1, y1)
    
    # Condition 2: t=999, y=4
    t2 = torch.full((batch_size,), 999, dtype=torch.long, device=device)
    y2 = torch.full((batch_size,), 4, dtype=torch.long, device=device)
    output2 = model(x, t2, y2)
    
    # Outputs should be different (conditioning is working)
    diff = (output1 - output2).abs().mean().item()
    assert diff > 1e-3, f"Outputs should differ with different conditioning, but diff={diff:.2e}"
    
    print("✓ DiT conditioning effect test passed!")
    print(f"  - Different conditioning produces different outputs")
    print(f"  - Mean absolute difference: {diff:.4f}")
    return True


def test_dit_per_sample_gradients():
    """Test that per-sample gradients are computed correctly for DiT"""
    print("\n" + "="*60)
    print("Test 4: DiT Per-Sample Gradient Correctness")
    print("="*60)
    
    device = "cpu"
    batch_size = 3
    seq_len = 8
    hidden_dim = 64
    num_classes = 5
    
    # Create model
    torch.manual_seed(42)
    model = SimplifiedDiTModel(
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_layers=1,
        num_heads=2,
        num_classes=num_classes,
    ).to(device)
    model = ModuleValidator.fix(model)
    
    # Create sample data
    torch.manual_seed(123)
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    t = torch.randint(0, 100, (batch_size,), device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)
    target = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    # Method 1: Compute per-sample gradients using GradSampleModule
    gs_model = GradSampleModule(model, batch_first=True, loss_reduction="mean")
    gs_model.train()
    output_batch = gs_model(x, t, y)
    loss_batch = F.mse_loss(output_batch, target)
    loss_batch.backward()
    
    # Get per-sample gradients
    per_sample_grads = {}
    for name, param in gs_model.named_parameters():
        if param.requires_grad and hasattr(param, 'grad_sample'):
            per_sample_grads[name] = param.grad_sample.clone()
    
    # Method 2: Compute gradients for each sample individually
    individual_grads = {name: [] for name in per_sample_grads.keys()}
    
    for i in range(batch_size):
        # Reset model
        torch.manual_seed(42)
        model_single = SimplifiedDiTModel(
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            num_layers=1,
            num_heads=2,
            num_classes=num_classes,
        ).to(device)
        model_single = ModuleValidator.fix(model_single)
        model_single.train()
        
        # Single sample forward/backward
        x_single = x[i:i+1]
        t_single = t[i:i+1]
        y_single = y[i:i+1]
        target_single = target[i:i+1]
        
        output_single = model_single(x_single, t_single, y_single)
        loss_single = F.mse_loss(output_single, target_single)
        loss_single.backward()
        
        # Collect gradients - build name mapping
        for (name_batch, _), (name_single, param) in zip(gs_model.named_parameters(), model_single.named_parameters()):
            if param.requires_grad and param.grad is not None and name_batch in individual_grads:
                individual_grads[name_batch].append(param.grad.clone())
    
    # Compare: per-sample gradients should match individual gradients
    max_diff_overall = 0.0
    for name in per_sample_grads.keys():
        per_sample_grad_tensor = per_sample_grads[name]
        individual_grad_stacked = torch.stack(individual_grads[name], dim=0)
        
        # They should be close (allowing for numerical differences)
        # DiT is more complex with multiple inputs, so use slightly relaxed tolerance
        max_diff = (per_sample_grad_tensor - individual_grad_stacked).abs().max().item()
        max_diff_overall = max(max_diff_overall, max_diff)
        
        assert max_diff < 5e-4, \
            f"Parameter {name}: per-sample grads differ from individual grads by {max_diff:.2e}"
    
    print("✓ DiT per-sample gradient correctness test passed!")
    print(f"  - Per-sample gradients match individual sample gradients")
    print(f"  - Maximum difference across all parameters: {max_diff_overall:.2e}")
    return True


def test_dit_determinism():
    """Test that DiT outputs are deterministic"""
    print("\n" + "="*60)
    print("Test 5: DiT Output Determinism")
    print("="*60)
    
    device = "cpu"
    batch_size = 4
    seq_len = 8
    hidden_dim = 64
    num_classes = 5
    
    # Create two identical models
    torch.manual_seed(42)
    model1 = SimplifiedDiTModel(
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_layers=1,
        num_heads=2,
        num_classes=num_classes,
    ).to(device)
    model1 = ModuleValidator.fix(model1)
    
    torch.manual_seed(42)
    model2 = SimplifiedDiTModel(
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_layers=1,
        num_heads=2,
        num_classes=num_classes,
    ).to(device)
    model2 = ModuleValidator.fix(model2)
    
    # Wrap with GradSampleModule
    gs_model1 = GradSampleModule(model1, batch_first=True, loss_reduction="mean")
    gs_model2 = GradSampleModule(model2, batch_first=True, loss_reduction="mean")
    
    # Create sample data
    torch.manual_seed(123)
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    t = torch.randint(0, 100, (batch_size,), device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)
    target = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    # Forward pass 1
    gs_model1.train()
    output1 = gs_model1(x, t, y)
    loss1 = F.mse_loss(output1, target)
    loss1.backward()
    
    # Forward pass 2
    gs_model2.train()
    output2 = gs_model2(x, t, y)
    loss2 = F.mse_loss(output2, target)
    loss2.backward()
    
    # Check outputs are identical
    assert torch.allclose(output1, output2, atol=1e-6), "Outputs should be identical"
    assert torch.allclose(loss1, loss2, atol=1e-6), "Losses should be identical"
    
    # Check grad_samples are identical
    for (name1, param1), (name2, param2) in zip(gs_model1.named_parameters(), gs_model2.named_parameters()):
        assert name1 == name2
        if param1.requires_grad:
            assert torch.allclose(param1.grad_sample, param2.grad_sample, atol=1e-5), \
                f"Parameter {name1} grad_sample should be identical"
    
    output_diff = (output1 - output2).abs().max().item()
    print("✓ DiT determinism test passed!")
    print(f"  - Outputs are identical: max diff = {output_diff:.2e}")
    print(f"  - All grad_samples are identical")
    return True


def main():
    """Run all DiT integration tests"""
    print("\n" + "="*70)
    print("DiT + OPACUS INTEGRATION TEST SUITE")
    print("Testing that multi-input conditional models work correctly")
    print("="*70)
    
    tests = [
        test_dit_forward_pass,
        test_dit_with_grad_sample_module,
        test_dit_conditioning_effect,
        test_dit_per_sample_gradients,
        test_dit_determinism,
    ]
    
    results = []
    for test_fn in tests:
        try:
            result = test_fn()
            results.append((test_fn.__name__, True, None))
        except Exception as e:
            results.append((test_fn.__name__, False, str(e)))
            print(f"✗ {test_fn.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, error in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {test_name}")
        if error:
            print(f"  Error: {error}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All DiT integration tests passed!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())

