#!/usr/bin/env python3
"""
Quick verification script for flash_fsdp_fuse_bk mode.

This script provides a fast sanity check that the fuse_bk implementation
is working correctly without running the full test suite.
"""

import torch
import torch.nn as nn

from opacus.grad_sample.utils import wrap_model


def quick_test():
    """Run quick verification tests."""
    
    print("="*70)
    print("Quick Verification: flash_fsdp_fuse_bk")
    print("="*70)
    print()
    
    # Test configuration
    B, T, D_in, D_hidden, D_out = 4, 32, 64, 128, 32
    max_grad_norm = 1.0
    
    # Simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(D_in, D_hidden)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(D_hidden, D_out)
        
        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))
    
    print("1. Testing flash_fsdp_fuse_bk mode...")
    print("-" * 70)
    
    # Create and wrap model
    torch.manual_seed(42)
    model = SimpleModel()
    
    wrapped = wrap_model(
        model,
        grad_sample_mode="flash_fsdp_fuse_bk",
        batch_first=True,
        loss_reduction="mean",
        max_grad_norm=max_grad_norm,
    )
    
    print(f"✓ Model wrapped successfully")
    print(f"  - Bookkeeping enabled: {wrapped.enable_bookkeeping}")
    print(f"  - Max grad norm: {wrapped.max_grad_norm}")
    print()
    
    # Create input
    torch.manual_seed(123)
    x = torch.randn(B, T, D_in)
    
    print("2. Running forward pass...")
    print("-" * 70)
    wrapped.zero_grad()
    y = wrapped(x)
    print(f"✓ Forward pass completed")
    print(f"  - Input shape: {x.shape}")
    print(f"  - Output shape: {y.shape}")
    print()
    
    print("3. Running backward pass...")
    print("-" * 70)
    loss = y.sum()
    loss.backward()
    print(f"✓ Backward pass completed")
    print(f"  - Loss: {loss.item():.6f}")
    print()
    
    print("4. Checking per-sample norms...")
    print("-" * 70)
    norms = wrapped.get_norm_sample()
    print(f"✓ Norms computed successfully")
    print(f"  - Norm shape: {norms.shape}")
    print(f"  - Norms: {norms}")
    print(f"  - Min norm: {norms.min().item():.6f}")
    print(f"  - Max norm: {norms.max().item():.6f}")
    print(f"  - Mean norm: {norms.mean().item():.6f}")
    print()
    
    print("5. Checking clipping coefficients...")
    print("-" * 70)
    coef = wrapped.get_clipping_coef()
    expected_coef = (max_grad_norm / (norms + 1e-6)).clamp(max=1.0)
    coef_match = torch.allclose(coef, expected_coef)
    print(f"✓ Clipping coefficients computed")
    print(f"  - Coefficients: {coef}")
    print(f"  - All <= 1.0: {(coef <= 1.0).all().item()}")
    print(f"  - Match expected: {coef_match}")
    print()
    
    print("6. Comparing with flash_fsdp_fuse (non-bookkeeping)...")
    print("-" * 70)
    
    # Create identical model with non-bookkeeping mode
    torch.manual_seed(42)
    model_fuse = SimpleModel()
    
    wrapped_fuse = wrap_model(
        model_fuse,
        grad_sample_mode="flash_fsdp_fuse",
        batch_first=True,
        loss_reduction="mean",
        max_grad_norm=max_grad_norm,
    )
    
    wrapped_fuse.zero_grad()
    y_fuse = wrapped_fuse(x.clone())
    loss_fuse = y_fuse.sum()
    loss_fuse.backward()
    norms_fuse = wrapped_fuse.get_norm_sample()
    
    output_match = torch.allclose(y, y_fuse, rtol=1e-5)
    norm_match = torch.allclose(norms, norms_fuse, rtol=1e-3, atol=1e-4)
    max_norm_diff = (norms - norms_fuse).abs().max().item()
    
    print(f"✓ Comparison completed")
    print(f"  - Output match: {output_match}")
    print(f"  - Norm match: {norm_match}")
    print(f"  - Max norm diff: {max_norm_diff:.6e}")
    print()
    
    print("="*70)
    print("VERIFICATION RESULTS")
    print("="*70)
    
    all_passed = all([
        norms.shape == (B,),
        (norms > 0).all(),
        (coef <= 1.0).all(),
        coef_match,
        output_match,
        norm_match,
    ])
    
    if all_passed:
        print("✅ All checks passed!")
        print()
        print("flash_fsdp_fuse_bk is working correctly:")
        print("  ✓ Forward/backward pass")
        print("  ✓ Norm computation")
        print("  ✓ Clipping coefficients")
        print("  ✓ Consistency with flash_fsdp_fuse")
    else:
        print("❌ Some checks failed!")
        print()
        if norms.shape != (B,):
            print(f"  ✗ Norm shape incorrect: {norms.shape} != {(B,)}")
        if not (norms > 0).all():
            print(f"  ✗ Some norms are non-positive")
        if not (coef <= 1.0).all():
            print(f"  ✗ Some clipping coefficients > 1.0")
        if not coef_match:
            print(f"  ✗ Clipping coefficients don't match expected")
        if not output_match:
            print(f"  ✗ Outputs don't match between modes")
        if not norm_match:
            print(f"  ✗ Norms don't match between modes (diff: {max_norm_diff:.6e})")
    
    print("="*70)
    print()
    
    return all_passed


if __name__ == "__main__":
    try:
        success = quick_test()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Verification failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

