#!/usr/bin/env python3
"""
Test DiT model with Flash/Ghost Clipping (Fast Gradient Clipping).

This tests the integration of multi-input DiT model with Opacus's fast gradient clipping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))

from opacus.validators import ModuleValidator
from opacus.grad_sample.grad_sample_module_fast_gradient_clipping import GradSampleModuleFastGradientClipping
from opacus.optimizers import DPOptimizerFastGradientClipping
from opacus.utils.fast_gradient_clipping_utils import DPLossFastGradientClipping


class SimplifiedDiTForFlashClip(nn.Module):
    """Simplified DiT model for testing flash clipping"""
    def __init__(self, seq_len=16, hidden_dim=128, num_classes=10):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, hidden_dim))
        
        # Simplified embedders
        self.timestep_embed = nn.Linear(1, hidden_dim)
        self.label_embed = nn.Embedding(num_classes, hidden_dim)
        
        # Single attention layer
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x, t, y):
        """
        Multi-input forward for DiT
        Args:
            x: (B, seq_len, hidden_dim)
            t: (B,) timesteps
            y: (B,) labels
        Returns:
            (B, seq_len, hidden_dim)
        """
        B = x.shape[0]
        
        # Input projection + pos embedding
        x = self.input_proj(x) + self.pos_embed
        
        # Conditioning
        if t.dim() == 1:
            t = t.unsqueeze(-1).float()
        t_emb = self.timestep_embed(t)  # (B, hidden_dim)
        y_emb = self.label_embed(y)      # (B, hidden_dim)
        c = t_emb + y_emb
        
        # Simple conditioning: add to each token
        x = x + c.unsqueeze(1)
        
        # Self-attention and MLP
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = x + self.mlp(x)
        
        # Output
        return self.output_proj(x)


def test_flash_clipping_basic():
    """Test basic flash clipping setup with simplified DiT"""
    print("\n" + "="*70)
    print("Test 1: Basic Flash Clipping with Simplified DiT")
    print("="*70)
    
    device = "cpu"
    batch_size = 4
    seq_len = 8
    hidden_dim = 64
    num_classes = 5
    
    # Create model
    model = SimplifiedDiTForFlashClip(
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    ).to(device)
    model = ModuleValidator.fix(model)
    
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Wrap with Flash Clipping GradSampleModule
    gs_model = GradSampleModuleFastGradientClipping(
        model,
        batch_first=True,
        loss_reduction="mean",
        max_grad_norm=1.0,
    )
    print("✓ Model wrapped with GradSampleModuleFastGradientClipping")
    
    # Create optimizer
    optimizer = DPOptimizerFastGradientClipping(
        optimizer=torch.optim.Adam(gs_model.parameters(), lr=1e-4),
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        expected_batch_size=batch_size,
        loss_reduction="mean",
    )
    print("✓ DPOptimizerFastGradientClipping created")
    
    # Create criterion (MSE loss for regression-like task)
    # Note: DPLossFastGradientClipping will set reduction="none" internally
    criterion = nn.MSELoss(reduction="mean")  # Will be changed to "none" by wrapper
    
    # Wrap loss with DPLossFastGradientClipping
    dp_loss_fn = DPLossFastGradientClipping(
        module=gs_model,
        optimizer=optimizer,
        criterion=criterion,
        loss_reduction="mean",
    )
    print("✓ DPLossFastGradientClipping wrapper created")
    
    # Create sample data
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    t = torch.randint(0, 100, (batch_size,), device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)
    target = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    print(f"\nInput shapes:")
    print(f"  x: {x.shape}")
    print(f"  t: {t.shape}")
    print(f"  y: {y.shape}")
    print(f"  target: {target.shape}")
    
    # Forward pass
    gs_model.train()
    output = gs_model(x, t, y)
    print(f"\n✓ Forward pass successful")
    print(f"  output: {output.shape}")
    
    # Compute per-sample loss
    # Key: flatten spatial dimensions before computing loss
    output_flat = output.reshape(batch_size, -1)
    target_flat = target.reshape(batch_size, -1)
    
    print(f"\nFlattened for loss computation:")
    print(f"  output_flat: {output_flat.shape}")
    print(f"  target_flat: {target_flat.shape}")
    
    # Use dp_loss_fn to compute loss
    # Need to compute per-sample loss manually or use criterion correctly
    loss_per_sample = F.mse_loss(output_flat, target_flat, reduction='none')
    loss_per_sample = loss_per_sample.mean(dim=1)  # Average over features, keep batch
    
    print(f"  loss_per_sample: {loss_per_sample.shape}")
    assert loss_per_sample.shape == (batch_size,), \
        f"Expected loss_per_sample shape ({batch_size},), got {loss_per_sample.shape}"
    
    # Manually create DPTensor for testing
    from opacus.utils.fast_gradient_clipping_utils import DPTensorFastGradientClipping
    dp_loss = DPTensorFastGradientClipping(
        module=gs_model,
        optimizer=optimizer,
        loss_per_sample=loss_per_sample,
        loss_reduction="mean",
    )
    
    print(f"\n✓ DPTensorFastGradientClipping created")
    print(f"  loss_per_sample shape: {loss_per_sample.shape}")
    print(f"  loss value: {dp_loss.item():.6f}")
    
    # Backward pass
    print(f"\n⚡ Running backward pass (ghost clipping)")
    dp_loss.backward()
    print(f"✓ Backward pass successful!")
    
    # Verify gradients exist
    grad_count = 0
    for name, param in gs_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"
            grad_count += 1
    
    print(f"✓ All {grad_count} parameters have gradients")
    
    # Step optimizer
    optimizer.step()
    optimizer.zero_grad()
    print(f"✓ Optimizer step successful")
    
    print("\n" + "="*70)
    print("✅ Test 1 PASSED")
    print("="*70)
    return True


def test_flash_clipping_with_loss_wrapper():
    """Test flash clipping using DPLossFastGradientClipping wrapper"""
    print("\n" + "="*70)
    print("Test 2: Flash Clipping with Loss Wrapper")
    print("="*70)
    
    device = "cpu"
    batch_size = 4
    seq_len = 8
    hidden_dim = 64
    num_classes = 5
    
    # Create model
    model = SimplifiedDiTForFlashClip(
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    ).to(device)
    model = ModuleValidator.fix(model)
    
    # Wrap with Flash Clipping
    gs_model = GradSampleModuleFastGradientClipping(
        model,
        batch_first=True,
        loss_reduction="mean",
        max_grad_norm=1.0,
    )
    
    optimizer = DPOptimizerFastGradientClipping(
        optimizer=torch.optim.Adam(gs_model.parameters(), lr=1e-4),
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        expected_batch_size=batch_size,
        loss_reduction="mean",
    )
    
    # Custom criterion that handles the multi-input model
    def custom_criterion(output, target):
        """
        Custom criterion that properly reduces spatial dimensions
        Args:
            output: (B, seq_len, hidden_dim)
            target: (B, seq_len, hidden_dim)
        Returns:
            loss_per_sample: (B,)
        """
        # Flatten spatial dims
        output_flat = output.reshape(output.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)
        
        # Compute per-sample MSE
        loss_per_sample = F.mse_loss(output_flat, target_flat, reduction='none')
        loss_per_sample = loss_per_sample.mean(dim=1)  # (B,)
        
        return loss_per_sample
    
    # Set reduction attribute for compatibility (will be set to "none" by wrapper)
    custom_criterion.reduction = "mean"
    
    # Wrap loss
    dp_loss_fn = DPLossFastGradientClipping(
        module=gs_model,
        optimizer=optimizer,
        criterion=custom_criterion,
        loss_reduction="mean",
    )
    
    print("✓ Setup complete with custom criterion")
    
    # Create data
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    t = torch.randint(0, 100, (batch_size,), device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)
    target = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    # Forward
    gs_model.train()
    output = gs_model(x, t, y)
    print(f"✓ Forward: output shape {output.shape}")
    
    # Compute loss using wrapper
    dp_loss = dp_loss_fn(output, target)
    print(f"✓ Loss computed: {dp_loss.item():.6f}")
    
    # Backward
    dp_loss.backward()
    print(f"✓ Backward successful")
    
    # Step
    optimizer.step()
    optimizer.zero_grad()
    print(f"✓ Optimizer step successful")
    
    print("\n" + "="*70)
    print("✅ Test 2 PASSED")
    print("="*70)
    return True


def test_flash_clipping_training_loop():
    """Test a complete training loop with flash clipping"""
    print("\n" + "="*70)
    print("Test 3: Complete Training Loop with Flash Clipping")
    print("="*70)
    
    device = "cpu"
    batch_size = 4
    seq_len = 8
    hidden_dim = 64
    num_classes = 5
    num_steps = 3
    
    # Setup
    model = SimplifiedDiTForFlashClip(seq_len, hidden_dim, num_classes).to(device)
    model = ModuleValidator.fix(model)
    
    gs_model = GradSampleModuleFastGradientClipping(
        model, batch_first=True, loss_reduction="mean", max_grad_norm=1.0
    )
    
    optimizer = DPOptimizerFastGradientClipping(
        optimizer=torch.optim.Adam(gs_model.parameters(), lr=1e-3),
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        expected_batch_size=batch_size,
        loss_reduction="mean",
    )
    
    def criterion(output, target):
        output_flat = output.reshape(output.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)
        loss = F.mse_loss(output_flat, target_flat, reduction='none')
        return loss.mean(dim=1)
    criterion.reduction = "mean"  # Will be changed to "none" by DPLossFastGradientClipping
    
    dp_loss_fn = DPLossFastGradientClipping(gs_model, optimizer, criterion, "mean")
    
    print(f"Training for {num_steps} steps...")
    losses = []
    
    for step in range(num_steps):
        # Generate data
        x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        t = torch.randint(0, 100, (batch_size,), device=device)
        y = torch.randint(0, num_classes, (batch_size,), device=device)
        target = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        
        # Train step
        gs_model.train()
        output = gs_model(x, t, y)
        dp_loss = dp_loss_fn(output, target)
        
        dp_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        loss_val = dp_loss.item()
        losses.append(loss_val)
        print(f"  Step {step+1}/{num_steps}: loss = {loss_val:.6f}")
    
    print(f"\n✓ Training completed successfully")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Loss change: {losses[-1] - losses[0]:.6f}")
    
    print("\n" + "="*70)
    print("✅ Test 3 PASSED")
    print("="*70)
    return True


def main():
    """Run all flash clipping tests"""
    print("\n" + "="*70)
    print("DiT + FLASH CLIPPING TEST SUITE")
    print("Testing multi-input models with Fast Gradient Clipping")
    print("="*70)
    
    tests = [
        test_flash_clipping_basic,
        test_flash_clipping_with_loss_wrapper,
        test_flash_clipping_training_loop,
    ]
    
    results = []
    for test_fn in tests:
        try:
            result = test_fn()
            results.append((test_fn.__name__, True, None))
        except Exception as e:
            results.append((test_fn.__name__, False, str(e)))
            print(f"\n✗ {test_fn.__name__} FAILED: {e}")
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
        print("\n🎉 All flash clipping tests passed!")
        print("\nKey Insight:")
        print("  For multi-dimensional outputs (B, seq_len, hidden_dim),")
        print("  flatten to (B, features) before computing loss")
        print("  to ensure loss_per_sample has shape (B,)")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())

