#!/usr/bin/env python3
"""
Integration test for Conv2d with bookkeeping in the DiT model.

This test verifies that:
1. Bookkeeping cache is populated for Conv2d layers
2. populate_clipped_gradients() works correctly with Conv2d
3. End-to-end gradient computation produces correct results
"""

import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))

import torch
import torch.nn as nn

from opacus.grad_sample import GradSampleModuleFastGradientClipping
from opacus.optimizers import DPOptimizerFastGradientClipping
from opacus.utils.fast_gradient_clipping_utils import DPLossFastGradientClipping


def test_simple_conv_bookkeeping():
    """Test bookkeeping with a simple Conv2d model."""
    print("\n" + "="*80)
    print("TEST 1: Simple Conv2d with Bookkeeping")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create a simple model with Conv2d
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10),
    ).to(device)
    
    # Wrap with bookkeeping enabled
    model = GradSampleModuleFastGradientClipping(
        model,
        use_flash_clipping=True,
        use_ghost_clipping=True,
        enable_fastdp_bookkeeping=True,
        loss_reduction="mean",
    )
    
    # Create optimizer
    base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = DPOptimizerFastGradientClipping(
        optimizer=base_optimizer,
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        expected_batch_size=4,
        loss_reduction="mean",
    )
    
    # Create loss function
    def criterion(predicted, target):
        """Criterion for per-sample loss."""
        return nn.functional.cross_entropy(predicted, target, reduction='none')
    
    criterion.reduction = "mean"
    
    dp_loss = DPLossFastGradientClipping(
        model,
        optimizer,
        criterion,
        loss_reduction="mean",
    )
    
    # Create dummy data
    images = torch.randn(4, 3, 8, 8, device=device)
    labels = torch.randint(0, 10, (4,), device=device)
    
    # Forward pass
    outputs = model(images)
    
    # Compute loss
    loss = dp_loss(outputs, labels)
    
    # Note: For bookkeeping, the cache is populated during backward and then
    # immediately consumed by populate_clipped_gradients (which is called inside
    # loss.backward()). After that, the cache is cleared.
    # So we can't directly check the cache after backward.
    
    # Backward pass (this populates cache, computes clipped gradients, and clears cache)
    print("Performing backward pass...")
    loss.backward()
    
    # The cache should be cleared after backward (bookkeeping consumes it immediately)
    print(f"Cache length after backward: {len(model._bk_cache)}")
    assert len(model._bk_cache) == 0, "Cache should be cleared after bookkeeping backward"
    
    # Perform optimizer step
    print("Performing optimizer step...")
    optimizer.step()
    
    # Check that gradients were populated
    conv_layers = [m for m in model._module.modules() if isinstance(m, nn.Conv2d)]
    assert len(conv_layers) >= 2, "Should have at least 2 Conv2d layers"
    for i, conv in enumerate(conv_layers):
        assert conv.weight.grad is not None, f"Conv layer {i} weight should have gradient"
        if conv.bias is not None:
            assert conv.bias.grad is not None, f"Conv layer {i} bias should have gradient"
        print(f"✓ Conv layer {i}: weight grad shape = {conv.weight.grad.shape}")
    
    print("✅ TEST 1 PASSED: Conv2d bookkeeping works correctly\n")
    return True


def test_conv_bookkeeping_correctness():
    """
    Test that bookkeeping produces the same gradients as standard two-pass approach.
    """
    print("\n" + "="*80)
    print("TEST 2: Conv2d Bookkeeping Correctness")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    
    # Create a simple model
    model_bk = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8 * 8 * 8, 10),
    ).to(device)
    
    # Clone for comparison
    model_std = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8 * 8 * 8, 10),
    ).to(device)
    
    # Copy weights
    model_std.load_state_dict(model_bk.state_dict())
    
    # Wrap with bookkeeping
    model_bk = GradSampleModuleFastGradientClipping(
        model_bk,
        use_flash_clipping=True,
        use_ghost_clipping=True,
        enable_fastdp_bookkeeping=True,
        loss_reduction="mean",
    )
    
    # Wrap without bookkeeping (standard two-pass)
    model_std = GradSampleModuleFastGradientClipping(
        model_std,
        use_flash_clipping=True,
        use_ghost_clipping=True,
        enable_fastdp_bookkeeping=False,
        loss_reduction="mean",
    )
    
    # Create optimizers
    opt_bk = DPOptimizerFastGradientClipping(
        torch.optim.SGD(model_bk.parameters(), lr=0.1),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        expected_batch_size=4,
        loss_reduction="mean",
    )
    
    opt_std = DPOptimizerFastGradientClipping(
        torch.optim.SGD(model_std.parameters(), lr=0.1),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        expected_batch_size=4,
        loss_reduction="mean",
    )
    
    # Create loss functions (separate instances for each model)
    def criterion_bk(predicted, target):
        """Criterion for per-sample loss (bookkeeping model)."""
        return nn.functional.cross_entropy(predicted, target, reduction='none')
    criterion_bk.reduction = "mean"
    
    def criterion_std(predicted, target):
        """Criterion for per-sample loss (standard model)."""
        return nn.functional.cross_entropy(predicted, target, reduction='none')
    criterion_std.reduction = "mean"
    
    dp_loss_bk = DPLossFastGradientClipping(model_bk, opt_bk, criterion_bk, loss_reduction="mean")
    dp_loss_std = DPLossFastGradientClipping(model_std, opt_std, criterion_std, loss_reduction="mean")
    
    # Create same data
    torch.manual_seed(123)
    images = torch.randn(4, 3, 8, 8, device=device)
    labels = torch.randint(0, 10, (4,), device=device)
    
    # Forward and backward with bookkeeping
    outputs_bk = model_bk(images)
    loss_bk = dp_loss_bk(outputs_bk, labels)
    loss_bk.backward()
    opt_bk.step()
    
    # Forward and backward without bookkeeping
    outputs_std = model_std(images)
    loss_std = dp_loss_std(outputs_std, labels)
    loss_std.backward()
    opt_std.step()
    
    # Compare gradients
    print("Comparing gradients...")
    params_bk = list(model_bk._module.parameters())
    params_std = list(model_std._module.parameters())
    
    max_diff = 0.0
    for i, (p_bk, p_std) in enumerate(zip(params_bk, params_std)):
        if p_bk.grad is not None and p_std.grad is not None:
            diff = (p_bk.grad - p_std.grad).abs().max().item()
            rel_diff = diff / (p_std.grad.abs().max().item() + 1e-8)
            max_diff = max(max_diff, rel_diff)
            print(f"  Param {i}: max abs diff = {diff:.2e}, max rel diff = {rel_diff:.2e}")
            
            # Check that gradients are close
            torch.testing.assert_close(
                p_bk.grad,
                p_std.grad,
                rtol=1e-5,
                atol=1e-7,
                msg=f"Gradient mismatch for parameter {i}",
            )
    
    print(f"Max relative difference: {max_diff:.2e}")
    print("✅ TEST 2 PASSED: Bookkeeping produces correct gradients\n")
    return True


def main():
    print("\n" + "#"*80)
    print("Conv2d Bookkeeping Integration Tests")
    print("#"*80)
    
    try:
        test_simple_conv_bookkeeping()
        test_conv_bookkeeping_correctness()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80 + "\n")
        return 0
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ TEST FAILED: {e}")
        print("="*80 + "\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

