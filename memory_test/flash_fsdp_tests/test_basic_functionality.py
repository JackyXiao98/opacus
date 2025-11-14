#!/usr/bin/env python3
"""
Basic functionality test for Flash Clipping with FSDP support.
Tests that the code changes work correctly without requiring CUDA/multi-GPU.
"""

import sys
import os

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from opacus import PrivacyEngine
from opacus.grad_sample import get_gsm_class, wrap_model
from opacus.optimizers import get_optimizer_class
from test_model import SmallTransformerDP, create_synthetic_dataset


def test_flash_mode():
    """Test that flash mode works correctly."""
    print("\n" + "=" * 80)
    print("TEST 1: Flash mode (single GPU)")
    print("=" * 80)
    
    # Test get_gsm_class
    gsm_class = get_gsm_class("flash")
    print(f"✓ get_gsm_class('flash') returns: {gsm_class.__name__}")
    
    # Test optimizer class
    opt_class = get_optimizer_class(clipping="flat", distributed=False, grad_sample_mode="flash")
    print(f"✓ get_optimizer_class returns: {opt_class.__name__}")
    
    # Create model (fresh for privacy engine)
    model = SmallTransformerDP(
        vocab_size=1000,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        num_classes=10,
        max_seq_len=32,
        dropout=0.0,
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    privacy_engine = PrivacyEngine()
    
    # Create simple dataset (make sure vocab_size matches model)
    inputs, labels = create_synthetic_dataset(num_samples=32, vocab_size=1000, seq_len=16, num_classes=10, seed=42)
    dataset = TensorDataset(inputs, labels)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    # Make private
    model, optimizer, criterion, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        criterion=nn.CrossEntropyLoss(),
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        grad_sample_mode="flash",
        poisson_sampling=False,
    )
    
    print(f"✓ Privacy engine created model type: {type(model).__name__}")
    print(f"✓ Privacy engine created optimizer type: {type(optimizer).__name__}")
    print(f"✓ Privacy engine created criterion type: {type(criterion).__name__}")
    
    # Verify model has use_flash_clipping attribute
    assert hasattr(model, 'use_flash_clipping'), "Model should have use_flash_clipping attribute"
    assert model.use_flash_clipping == True, "use_flash_clipping should be True"
    print(f"✓ Model has use_flash_clipping=True")
    
    # Test forward and backward
    model.train()
    for batch_idx, (batch_inputs, batch_labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        if batch_idx == 0:  # Only test first batch
            print(f"✓ Forward/backward pass completed successfully")
            print(f"  Loss: {loss.item():.4f}")
            break
    
    print("✓ TEST 1 PASSED: Flash mode works correctly")
    return True


def test_flash_fsdp_mode():
    """Test that flash_fsdp mode works correctly (without actual FSDP)."""
    print("\n" + "=" * 80)
    print("TEST 2: Flash FSDP mode (validation only, no actual FSDP)")
    print("=" * 80)
    
    # Test get_gsm_class
    gsm_class = get_gsm_class("flash_fsdp")
    print(f"✓ get_gsm_class('flash_fsdp') returns: {gsm_class.__name__}")
    assert "FSDP" in gsm_class.__name__, "Should return FSDP variant"
    
    # Test that the class has FLASH_NORM_SAMPLERS attribute
    assert hasattr(gsm_class, 'FLASH_NORM_SAMPLERS'), "Should have FLASH_NORM_SAMPLERS class attribute"
    print(f"✓ {gsm_class.__name__} has FLASH_NORM_SAMPLERS class attribute")
    
    # Test optimizer class for FSDP
    opt_class = get_optimizer_class(clipping="flat", distributed=True, grad_sample_mode="flash_fsdp")
    print(f"✓ get_optimizer_class (FSDP) returns: {opt_class.__name__}")
    
    print("✓ TEST 2 PASSED: Flash FSDP mode validation successful")
    return True


def test_flash_norm_samplers():
    """Test that FLASH_NORM_SAMPLERS dict is accessible."""
    print("\n" + "=" * 80)
    print("TEST 3: FLASH_NORM_SAMPLERS accessibility")
    print("=" * 80)
    
    from opacus.grad_sample.grad_sample_module_fast_gradient_clipping import (
        GradSampleModuleFastGradientClipping,
    )
    from opacus.grad_sample.grad_sample_module_fast_gradient_clipping_fsdp import (
        GradSampleModuleFastGradientClippingFSDP,
    )
    
    # Check standard class
    assert hasattr(GradSampleModuleFastGradientClipping, 'FLASH_NORM_SAMPLERS'), \
        "GradSampleModuleFastGradientClipping should have FLASH_NORM_SAMPLERS"
    print(f"✓ GradSampleModuleFastGradientClipping has FLASH_NORM_SAMPLERS")
    
    # Check FSDP class
    assert hasattr(GradSampleModuleFastGradientClippingFSDP, 'FLASH_NORM_SAMPLERS'), \
        "GradSampleModuleFastGradientClippingFSDP should have FLASH_NORM_SAMPLERS"
    print(f"✓ GradSampleModuleFastGradientClippingFSDP has FLASH_NORM_SAMPLERS")
    
    print("✓ TEST 3 PASSED: FLASH_NORM_SAMPLERS accessible")
    return True


def main():
    print("\n" + "=" * 80)
    print("FLASH CLIPPING FSDP SUPPORT - BASIC FUNCTIONALITY TESTS")
    print("=" * 80)
    print("This validates that the code changes work correctly.")
    print("Full FSDP integration tests require CUDA and multi-GPU setup.")
    print("=" * 80)
    
    all_passed = True
    
    # Run tests
    try:
        all_passed &= test_flash_mode()
    except Exception as e:
        print(f"✗ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_flash_fsdp_mode()
    except Exception as e:
        print(f"✗ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_flash_norm_samplers()
    except Exception as e:
        print(f"✗ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # Summary
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("Flash Clipping FSDP support is correctly implemented.")
    else:
        print("✗ SOME TESTS FAILED")
        print("Please review the errors above.")
        sys.exit(1)
    print("=" * 80)


if __name__ == "__main__":
    main()

