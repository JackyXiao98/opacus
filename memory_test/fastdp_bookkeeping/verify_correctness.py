#!/usr/bin/env python3
"""
Verification Script for FastDP Bookkeeping (BK) Implementation

This script verifies that the Bookkeeping (BK) optimization produces
numerically identical gradients to the standard 2-pass Ghost Clipping approach.

The test:
1. Creates a simple model (Linear layers)
2. Runs forward + backward with standard Ghost Clipping (2 passes)
3. Runs forward + backward with Bookkeeping (1 pass)
4. Compares gradients with torch.allclose()

Author: AI Research Engineer
Date: 2025-11-11
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from opacus.grad_sample.grad_sample_module_fast_gradient_clipping import (
    GradSampleModuleFastGradientClipping,
)
from opacus.optimizers import DPOptimizerFastGradientClipping
from opacus.utils.fast_gradient_clipping_utils import DPLossFastGradientClipping


class SimpleModel(nn.Module):
    """Simple model for testing with Linear layers"""
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


class SequenceModel(nn.Module):
    """Model with 3D tensors (batch, sequence, features) to test sequence handling"""
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=32, seq_len=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.seq_len = seq_len
    
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_model_gradients(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract gradients from model as a dictionary"""
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.clone().detach()
        else:
            grads[name] = None
    return grads


def compare_gradients(
    grads1: Dict[str, torch.Tensor],
    grads2: Dict[str, torch.Tensor],
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> Tuple[bool, Dict[str, float]]:
    """
    Compare two gradient dictionaries
    
    Returns:
        (all_close, max_diffs): all_close is True if all gradients match,
                                max_diffs contains max absolute difference per parameter
    """
    all_close = True
    max_diffs = {}
    
    for name in grads1.keys():
        g1 = grads1[name]
        g2 = grads2[name]
        
        if g1 is None and g2 is None:
            max_diffs[name] = 0.0
            continue
        
        if g1 is None or g2 is None:
            print(f"  ❌ {name}: One gradient is None, other is not")
            all_close = False
            max_diffs[name] = float('inf')
            continue
        
        # Check if close
        is_close = torch.allclose(g1, g2, rtol=rtol, atol=atol)
        max_diff = torch.max(torch.abs(g1 - g2)).item()
        max_diffs[name] = max_diff
        
        if not is_close:
            print(f"  ❌ {name}: max_diff = {max_diff:.2e}, rtol={rtol}, atol={atol}")
            print(f"      g1 norm: {g1.norm().item():.6f}, g2 norm: {g2.norm().item():.6f}")
            all_close = False
        else:
            print(f"  ✅ {name}: max_diff = {max_diff:.2e} (PASS)")
    
    return all_close, max_diffs


def run_standard_ghost_clipping(
    model: nn.Module,
    data: torch.Tensor,
    labels: torch.Tensor,
    max_grad_norm: float = 1.0,
    batch_size: int = 4,
    use_triton: bool = False,
) -> Dict[str, torch.Tensor]:
    """Run standard Ghost Clipping (2-pass) and return gradients"""
    
    # Wrap model with GradSampleModule (bookkeeping=False)
    wrapped_model = GradSampleModuleFastGradientClipping(
        model,
        batch_first=True,
        max_grad_norm=max_grad_norm,
        use_ghost_clipping=True,
        use_triton=use_triton,
        loss_reduction="mean",
        enable_fastdp_bookkeeping=False,  # Standard 2-pass mode
    )
    
    # Create optimizer
    base_optimizer = torch.optim.SGD(wrapped_model.parameters(), lr=0.01)
    optimizer = DPOptimizerFastGradientClipping(
        optimizer=base_optimizer,
        noise_multiplier=0.0,  # No noise for correctness test
        max_grad_norm=max_grad_norm,
        expected_batch_size=batch_size,
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
    
    # Forward pass
    optimizer.zero_grad()
    outputs = wrapped_model(data)
    
    # Compute per-sample loss
    if labels.dim() == 2:
        # Sequence labeling case - need to handle per-sequence loss
        # outputs: [B, T, vocab], labels: [B, T]
        loss_tensor = dp_loss(outputs.view(-1, outputs.shape[-1]), labels.view(-1), shape=outputs.shape[:2])
    else:
        loss_tensor = dp_loss(outputs, labels)
    
    # Backward pass (2 passes internally)
    loss_tensor.backward()
    
    # Extract gradients
    grads = get_model_gradients(wrapped_model._module)
    
    return grads


def run_bookkeeping_clipping(
    model: nn.Module,
    data: torch.Tensor,
    labels: torch.Tensor,
    max_grad_norm: float = 1.0,
    batch_size: int = 4,
    use_triton: bool = False,
) -> Dict[str, torch.Tensor]:
    """Run Bookkeeping (BK) clipping (1-pass) and return gradients"""
    
    # Wrap model with GradSampleModule (bookkeeping=True)
    wrapped_model = GradSampleModuleFastGradientClipping(
        model,
        batch_first=True,
        max_grad_norm=max_grad_norm,
        use_ghost_clipping=True,
        use_triton=use_triton,
        loss_reduction="mean",
        enable_fastdp_bookkeeping=True,  # Bookkeeping mode!
    )
    
    # Create optimizer
    base_optimizer = torch.optim.SGD(wrapped_model.parameters(), lr=0.01)
    optimizer = DPOptimizerFastGradientClipping(
        optimizer=base_optimizer,
        noise_multiplier=0.0,  # No noise for correctness test
        max_grad_norm=max_grad_norm,
        expected_batch_size=batch_size,
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
    
    # Forward pass
    optimizer.zero_grad()
    outputs = wrapped_model(data)
    
    # Compute per-sample loss
    if labels.dim() == 2:
        # Sequence labeling case - need to handle per-sequence loss
        # outputs: [B, T, vocab], labels: [B, T]
        loss_tensor = dp_loss(outputs.view(-1, outputs.shape[-1]), labels.view(-1), shape=outputs.shape[:2])
    else:
        loss_tensor = dp_loss(outputs, labels)
    
    # Backward pass (1 pass + manual gradient computation)
    loss_tensor.backward()
    
    # Extract gradients
    grads = get_model_gradients(wrapped_model._module)
    
    return grads


def test_2d_case(use_triton=False):
    """Test with 2D inputs (standard batch of vectors)"""
    print("\n" + "="*80)
    print(f"TEST 1: 2D Case (Batch of Vectors) - use_triton={use_triton}")
    print("="*80)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create data
    batch_size = 8
    input_dim = 128
    hidden_dim = 256
    output_dim = 10
    
    data = torch.randn(batch_size, input_dim)
    labels = torch.randint(0, output_dim, (batch_size,))
    
    # Test with standard Ghost Clipping
    print("\n--- Running Standard Ghost Clipping (2-pass) ---")
    model1 = SimpleModel(input_dim, hidden_dim, output_dim)
    grads_standard = run_standard_ghost_clipping(
        model1, data, labels, max_grad_norm=1.0, batch_size=batch_size, use_triton=use_triton
    )
    
    # Test with Bookkeeping
    print("\n--- Running Bookkeeping (1-pass) ---")
    model2 = SimpleModel(input_dim, hidden_dim, output_dim)
    # Copy weights to ensure same initialization
    model2.load_state_dict(model1.state_dict())
    grads_bookkeeping = run_bookkeeping_clipping(
        model2, data, labels, max_grad_norm=1.0, batch_size=batch_size, use_triton=use_triton
    )
    
    # Compare gradients
    print("\n--- Comparing Gradients ---")
    all_close, max_diffs = compare_gradients(grads_standard, grads_bookkeeping, rtol=1e-4, atol=1e-5)
    
    if all_close:
        print("\n✅ TEST PASSED: All gradients match!")
        return True
    else:
        print("\n❌ TEST FAILED: Gradients do not match!")
        return False


def test_3d_case(use_triton=False):
    """Test with 3D inputs (batch, sequence_length, features)"""
    print("\n" + "="*80)
    print(f"TEST 2: 3D Case (Sequence Data) - use_triton={use_triton}")
    print("="*80)
    
    # Set random seed
    torch.manual_seed(123)
    np.random.seed(123)
    
    # Create sequence data
    batch_size = 4
    seq_len = 16
    input_dim = 64
    hidden_dim = 128
    output_dim = 32
    
    data = torch.randn(batch_size, seq_len, input_dim)
    labels = torch.randint(0, output_dim, (batch_size, seq_len))
    
    # Test with standard Ghost Clipping
    print("\n--- Running Standard Ghost Clipping (2-pass) ---")
    model1 = SequenceModel(input_dim, hidden_dim, output_dim, seq_len)
    grads_standard = run_standard_ghost_clipping(
        model1, data, labels, max_grad_norm=1.0, batch_size=batch_size, use_triton=use_triton
    )
    
    # Test with Bookkeeping
    print("\n--- Running Bookkeeping (1-pass) ---")
    model2 = SequenceModel(input_dim, hidden_dim, output_dim, seq_len)
    model2.load_state_dict(model1.state_dict())
    grads_bookkeeping = run_bookkeeping_clipping(
        model2, data, labels, max_grad_norm=1.0, batch_size=batch_size, use_triton=use_triton
    )
    
    # Compare gradients
    print("\n--- Comparing Gradients ---")
    all_close, max_diffs = compare_gradients(grads_standard, grads_bookkeeping, rtol=1e-4, atol=1e-5)
    
    if all_close:
        print("\n✅ TEST PASSED: All gradients match!")
        return True
    else:
        print("\n❌ TEST FAILED: Gradients do not match!")
        return False


def test_different_clipping_norms(use_triton=False):
    """Test with different gradient clipping norms"""
    print("\n" + "="*80)
    print(f"TEST 3: Different Clipping Norms - use_triton={use_triton}")
    print("="*80)
    
    batch_size = 8
    input_dim = 64
    hidden_dim = 128
    output_dim = 10
    
    test_norms = [0.5, 1.0, 2.0, 5.0]
    all_passed = True
    
    for norm in test_norms:
        print(f"\n--- Testing with max_grad_norm={norm} ---")
        
        torch.manual_seed(42)
        data = torch.randn(batch_size, input_dim)
        labels = torch.randint(0, output_dim, (batch_size,))
        
        model1 = SimpleModel(input_dim, hidden_dim, output_dim)
        grads_standard = run_standard_ghost_clipping(
            model1, data, labels, max_grad_norm=norm, batch_size=batch_size, use_triton=use_triton
        )
        
        model2 = SimpleModel(input_dim, hidden_dim, output_dim)
        model2.load_state_dict(model1.state_dict())
        grads_bookkeeping = run_bookkeeping_clipping(
            model2, data, labels, max_grad_norm=norm, batch_size=batch_size, use_triton=use_triton
        )
        
        all_close, max_diffs = compare_gradients(grads_standard, grads_bookkeeping, rtol=1e-4, atol=1e-5)
        
        if not all_close:
            all_passed = False
            print(f"❌ Failed for norm={norm}")
        else:
            print(f"✅ Passed for norm={norm}")
    
    return all_passed


def main():
    """Run all verification tests"""
    print("="*80)
    print("FastDP Bookkeeping (BK) Correctness Verification")
    print("="*80)
    print("\nThis script verifies that Bookkeeping produces identical gradients")
    print("to standard Ghost Clipping (within numerical tolerance).")
    
    # Check CUDA availability
    device_info = f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}"
    print(f"\n{device_info}")
    
    # Run tests without Triton first (always available)
    print("\n" + "="*80)
    print("RUNNING TESTS WITHOUT TRITON ACCELERATION")
    print("="*80)
    
    results = []
    results.append(("2D Case", test_2d_case(use_triton=False)))
    results.append(("3D Case", test_3d_case(use_triton=False)))
    results.append(("Different Norms", test_different_clipping_norms(use_triton=False)))
    
    # Try tests with Triton if available
    try:
        import triton
        print("\n" + "="*80)
        print("RUNNING TESTS WITH TRITON ACCELERATION")
        print("="*80)
        results.append(("2D Case (Triton)", test_2d_case(use_triton=True)))
        results.append(("3D Case (Triton)", test_3d_case(use_triton=True)))
        results.append(("Different Norms (Triton)", test_different_clipping_norms(use_triton=True)))
    except ImportError:
        print("\n⚠️  Triton not available, skipping Triton tests")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*80)
        print("\nThe Bookkeeping implementation is numerically correct.")
        return 0
    else:
        print("\n" + "="*80)
        print("❌ SOME TESTS FAILED")
        print("="*80)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

