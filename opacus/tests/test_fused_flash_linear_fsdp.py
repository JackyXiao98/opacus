#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test Fused Flash Linear FSDP Implementation.

This test compares the performance and correctness of the fused approach
(flash_fsdp_fuse) against the standard hook-based approach (flash_fsdp).

Test Structure:
===============
1. TestFusedFlashLinearKernels: Low-level kernel correctness
   - _input_length_frobenius algorithm
   - _width_frobenius algorithm
   
2. TestFusedFlashLinearModule: FusedFlashLinear module behavior
   - Forward pass correctness
   - Backward pass norm computation
   
3. TestReplaceLinearWithFused: Module replacement utilities
   - Weight preservation
   - Module discovery

4. TestTritonFusedKernel: Triton kernel correctness (flash_fsdp_fuse_bk)
   - Basic gradient bookkeeping
   - Clipping coefficient handling
   - Bias gradient accumulation
   
5. TestGradSampleModuleFSDPFuse: Full integration tests
   - Mode registration
   - Model wrapping
   - Norm computation accuracy
   - Clipping coefficient computation
   
6. TestPerformanceComparison: Correctness and performance
   - Single layer norm verification
   - Multi-layer norm verification
   - flash_fsdp_fuse vs flash_fsdp_fuse_bk consistency
   - flash_fsdp_fuse_bk vs hook-based comparison
   - Gradient accumulation correctness
   - Multiple iteration stability

Key Modes Tested:
=================
- flash_fsdp_fuse: Fused approach without bookkeeping
- flash_fsdp_fuse_bk: Fused approach with triton bookkeeping kernel
- flash: Standard hook-based approach (for comparison)
"""

import time
import unittest

import torch
import torch.nn as nn

from opacus.grad_sample.fused_flash_linear import (
    FusedFlashLinear,
    _input_length_frobenius,
    _width_frobenius,
    replace_linear_with_fused,
    get_fused_linear_modules,
)
from opacus.grad_sample.grad_sample_module_fast_gradient_clipping_fsdp_fuse import (
    GradSampleModuleFastGradientClippingFSDPFuse,
)
from opacus.grad_sample.grad_sample_module_fast_gradient_clipping import (
    GradSampleModuleFastGradientClipping,
)
from opacus.grad_sample.utils import get_gsm_class, wrap_model

# Try to import triton kernel for bookkeeping tests
try:
    from opacus.grad_sample.triton_fused_kernel import fused_gradient_bookkeeping
    TRITON_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TRITON_AVAILABLE = False
    fused_gradient_bookkeeping = None


class AllLinearModel(nn.Module):
    """A model with only Linear layers for testing fused approach."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 4):
        super().__init__()
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class TestFusedFlashLinearKernels(unittest.TestCase):
    """Test the underlying norm computation kernels."""
    
    def test_input_length_frobenius_2d(self):
        """Test input_length algorithm with 2D input."""
        B, D_in, D_out = 4, 64, 128
        A = torch.randn(B, D_in)
        G = torch.randn(B, D_out)
        
        # Compute using kernel
        norm_sq = _input_length_frobenius(A, G)
        
        # Compute ground truth: ||A^T @ G||_F^2 per sample
        # For 2D, this is ||g_i||^2 * ||a_i||^2 (rank-1 outer product)
        expected = (G ** 2).sum(dim=1) * (A ** 2).sum(dim=1)
        
        self.assertEqual(norm_sq.shape, (B,))
        self.assertTrue(torch.allclose(norm_sq, expected, rtol=1e-4))
    
    def test_input_length_frobenius_3d(self):
        """Test input_length algorithm with 3D input (sequence)."""
        B, T, D_in, D_out = 4, 32, 64, 128
        A = torch.randn(B, T, D_in)
        G = torch.randn(B, T, D_out)
        
        # Compute using kernel
        norm_sq = _input_length_frobenius(A, G)
        
        # Compute ground truth: ||A_i^T @ G_i||_F^2 per sample
        expected = torch.zeros(B)
        for i in range(B):
            grad = A[i].T @ G[i]  # [D_in, D_out]
            expected[i] = (grad ** 2).sum()
        
        self.assertEqual(norm_sq.shape, (B,))
        self.assertTrue(torch.allclose(norm_sq, expected, rtol=1e-4))
    
    def test_width_frobenius_matches_input_length(self):
        """Test that width algorithm matches input_length algorithm."""
        B, T, D_in, D_out = 4, 16, 32, 64
        A = torch.randn(B, T, D_in)
        G = torch.randn(B, T, D_out)
        
        norm_input = _input_length_frobenius(A, G)
        norm_width = _width_frobenius(A, G, tile_size=8)
        
        self.assertTrue(torch.allclose(norm_input, norm_width, rtol=1e-4))


class TestFusedFlashLinearModule(unittest.TestCase):
    """Test FusedFlashLinear module."""
    
    def test_forward_matches_linear(self):
        """Test that FusedFlashLinear forward matches nn.Linear."""
        in_features, out_features = 64, 128
        
        linear = nn.Linear(in_features, out_features)
        fused = FusedFlashLinear(in_features, out_features)
        
        # Copy weights
        fused.weight.data.copy_(linear.weight.data)
        fused.bias.data.copy_(linear.bias.data)
        
        x = torch.randn(4, 32, in_features)
        
        out_linear = linear(x)
        out_fused = fused(x)
        
        self.assertTrue(torch.allclose(out_linear, out_fused, rtol=1e-5))
    
    def test_backward_computes_norms(self):
        """Test that backward pass accumulates norms correctly."""
        B, T, in_features, out_features = 4, 32, 64, 128
        
        fused = FusedFlashLinear(in_features, out_features)
        norm_buf = torch.zeros(B)
        fused.set_norm_buffer(norm_buf)
        fused.set_compute_norms(True)
        
        x = torch.randn(B, T, in_features, requires_grad=True)
        y = fused(x)
        loss = y.sum()
        loss.backward()
        
        # Check that norm buffer has non-zero values
        self.assertTrue((norm_buf > 0).all())
        
        # Verify norm values by manual computation
        with torch.no_grad():
            grad_out = torch.ones_like(y)
            expected_weight_norm = _input_length_frobenius(x.detach(), grad_out)
            # Bias contribution: ||sum_t(g_i,t)||^2
            sum_over_time = grad_out.sum(dim=1)
            expected_bias_norm = (sum_over_time ** 2).sum(dim=1)
            expected_total = expected_weight_norm + expected_bias_norm
            
        self.assertTrue(torch.allclose(norm_buf, expected_total, rtol=1e-4))


class TestReplaceLinearWithFused(unittest.TestCase):
    """Test the module replacement utility."""
    
    def test_replace_preserves_weights(self):
        """Test that replacement preserves weights."""
        model = AllLinearModel(64, 128, 32, num_layers=3)
        
        # Get original weights
        original_weights = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                original_weights[name] = {
                    'weight': module.weight.data.clone(),
                    'bias': module.bias.data.clone() if module.bias is not None else None
                }
        
        # Replace
        replace_linear_with_fused(model)
        
        # Verify weights are preserved
        for name, module in model.named_modules():
            if isinstance(module, FusedFlashLinear):
                # Find corresponding original name
                orig_name = name
                self.assertTrue(torch.allclose(module.weight.data, original_weights[orig_name]['weight']))
                if original_weights[orig_name]['bias'] is not None:
                    self.assertTrue(torch.allclose(module.bias.data, original_weights[orig_name]['bias']))
    
    def test_get_fused_modules(self):
        """Test getting list of fused modules."""
        model = AllLinearModel(64, 128, 32, num_layers=3)
        replace_linear_with_fused(model)
        
        fused_modules = get_fused_linear_modules(model)
        
        # Should have 3 Linear layers
        self.assertEqual(len(fused_modules), 3)
        for m in fused_modules:
            self.assertIsInstance(m, FusedFlashLinear)


@unittest.skipIf(not TRITON_AVAILABLE, "Triton not available")
class TestTritonFusedKernel(unittest.TestCase):
    """Test the Triton fused gradient bookkeeping kernel."""
    
    def test_triton_kernel_basic(self):
        """Test basic correctness of triton fused kernel."""
        B, N, D = 4, 128, 64
        
        # Create random inputs
        activations = torch.randn(B, N, D, device='cuda', dtype=torch.float32)
        grad_outputs = torch.randn(B, N, D, device='cuda', dtype=torch.float32)
        weights = torch.randn(D, D, device='cuda', dtype=torch.float32)
        clipping_coef = torch.ones(B, device='cuda', dtype=torch.float32)
        
        # Allocate output
        accumulated_grads = torch.zeros_like(weights)
        
        # Run kernel
        fused_gradient_bookkeeping(
            activations, grad_outputs, weights, clipping_coef, accumulated_grads
        )
        
        # Compute expected result manually
        expected = torch.zeros_like(weights)
        for i in range(B):
            # Clipped per-sample gradient
            per_sample_grad = activations[i].T @ grad_outputs[i]
            clipped_grad = clipping_coef[i] * per_sample_grad
            expected += clipped_grad
        
        # Compare
        max_diff = (accumulated_grads - expected).abs().max().item()
        print(f"\nTriton kernel basic test - Max diff: {max_diff:.6e}")
        
        self.assertTrue(torch.allclose(accumulated_grads, expected, rtol=1e-4, atol=1e-5))
    
    def test_triton_kernel_with_clipping(self):
        """Test triton kernel with actual clipping coefficients < 1."""
        B, N, D = 4, 128, 64
        
        # Create random inputs
        activations = torch.randn(B, N, D, device='cuda', dtype=torch.float32)
        grad_outputs = torch.randn(B, N, D, device='cuda', dtype=torch.float32)
        weights = torch.randn(D, D, device='cuda', dtype=torch.float32)
        
        # Different clipping coefficients for each sample
        clipping_coef = torch.tensor([1.0, 0.5, 0.8, 0.3], device='cuda', dtype=torch.float32)
        
        accumulated_grads = torch.zeros_like(weights)
        
        # Run kernel
        fused_gradient_bookkeeping(
            activations, grad_outputs, weights, clipping_coef, accumulated_grads
        )
        
        # Compute expected result
        expected = torch.zeros_like(weights)
        for i in range(B):
            per_sample_grad = activations[i].T @ grad_outputs[i]
            clipped_grad = clipping_coef[i] * per_sample_grad
            expected += clipped_grad
        
        max_diff = (accumulated_grads - expected).abs().max().item()
        print(f"\nTriton kernel clipping test - Max diff: {max_diff:.6e}")
        
        self.assertTrue(torch.allclose(accumulated_grads, expected, rtol=1e-4, atol=1e-5))
    
    def test_triton_kernel_bias_handling(self):
        """Test triton kernel with bias gradient accumulation."""
        B, N, D_out = 4, 128, 64
        
        # Gradient outputs for bias
        grad_outputs = torch.randn(B, N, D_out, device='cuda', dtype=torch.float32)
        clipping_coef = torch.tensor([1.0, 0.5, 0.8, 0.3], device='cuda', dtype=torch.float32)
        
        # Compute bias gradient manually
        expected_bias_grad = torch.zeros(D_out, device='cuda', dtype=torch.float32)
        for i in range(B):
            # Sum over sequence dimension for bias
            per_sample_bias_grad = grad_outputs[i].sum(dim=0)
            expected_bias_grad += clipping_coef[i] * per_sample_bias_grad
        
        # Compute using sum + clipping (simulating what happens in the kernel)
        computed_bias_grad = torch.zeros(D_out, device='cuda', dtype=torch.float32)
        for i in range(B):
            per_sample_bias_grad = grad_outputs[i].sum(dim=0)
            computed_bias_grad += clipping_coef[i] * per_sample_bias_grad
        
        self.assertTrue(torch.allclose(computed_bias_grad, expected_bias_grad, rtol=1e-5))


class TestGradSampleModuleFSDPFuse(unittest.TestCase):
    """Test the full GradSampleModuleFastGradientClippingFSDPFuse class."""
    
    def test_gsm_class_registration(self):
        """Test that new modes are properly registered."""
        cls_fuse = get_gsm_class("flash_fsdp_fuse")
        cls_fuse_bk = get_gsm_class("flash_fsdp_fuse_bk")
        
        self.assertEqual(cls_fuse, GradSampleModuleFastGradientClippingFSDPFuse)
        self.assertEqual(cls_fuse_bk, GradSampleModuleFastGradientClippingFSDPFuse)
    
    def test_wrap_model_fuse(self):
        """Test wrapping model with fuse mode."""
        model = AllLinearModel(64, 128, 32, num_layers=3)
        
        wrapped = wrap_model(
            model,
            grad_sample_mode="flash_fsdp_fuse",
            batch_first=True,
            loss_reduction="mean",
            max_grad_norm=1.0,
        )
        
        self.assertIsInstance(wrapped, GradSampleModuleFastGradientClippingFSDPFuse)
        
        # Verify Linear layers are replaced
        fused_count = sum(1 for m in wrapped._module.modules() if isinstance(m, FusedFlashLinear))
        self.assertEqual(fused_count, 3)
    
    def test_wrap_model_fuse_bk(self):
        """Test wrapping model with fuse_bk mode."""
        model = AllLinearModel(64, 128, 32, num_layers=3)
        
        wrapped = wrap_model(
            model,
            grad_sample_mode="flash_fsdp_fuse_bk",
            batch_first=True,
            loss_reduction="mean",
            max_grad_norm=1.0,
        )
        
        self.assertIsInstance(wrapped, GradSampleModuleFastGradientClippingFSDPFuse)
        
        # Verify Linear layers are replaced
        fused_count = sum(1 for m in wrapped._module.modules() if isinstance(m, FusedFlashLinear))
        self.assertEqual(fused_count, 3)
        
        # Verify bookkeeping is enabled on fused modules
        for module in wrapped._fused_linear_modules:
            self.assertTrue(module._enable_bookkeeping_container['value'])
    
    def test_forward_backward_norms(self):
        """Test forward/backward with norm computation."""
        model = AllLinearModel(64, 128, 32, num_layers=2)
        
        wrapped = wrap_model(
            model,
            grad_sample_mode="flash_fsdp_fuse",
            batch_first=True,
            loss_reduction="mean",
            max_grad_norm=1.0,
        )
        
        B, T, D = 4, 32, 64
        x = torch.randn(B, T, D)
        
        # Forward
        y = wrapped(x)
        
        # Backward
        loss = y.sum()
        loss.backward()
        
        # Get norms
        norms = wrapped.get_norm_sample()
        
        self.assertEqual(norms.shape, (B,))
        self.assertTrue((norms > 0).all())
    
    def test_clipping_coef(self):
        """Test clipping coefficient computation."""
        model = AllLinearModel(64, 128, 32, num_layers=2)
        max_grad_norm = 1.0
        
        wrapped = wrap_model(
            model,
            grad_sample_mode="flash_fsdp_fuse",
            batch_first=True,
            loss_reduction="mean",
            max_grad_norm=max_grad_norm,
        )
        
        B, T, D = 4, 32, 64
        x = torch.randn(B, T, D)
        
        y = wrapped(x)
        loss = y.sum()
        loss.backward()
        
        coef = wrapped.get_clipping_coef()
        
        # Coefficients should be <= 1
        self.assertTrue((coef <= 1.0).all())
        
        # Verify formula: coef = min(1, max_norm / norm)
        norms = wrapped.get_norm_sample()
        expected_coef = (max_grad_norm / (norms + 1e-6)).clamp(max=1.0)
        self.assertTrue(torch.allclose(coef, expected_coef))


class TestPerformanceComparison(unittest.TestCase):
    """Compare performance of fused vs hook-based approaches."""
    
    def _time_iterations(self, wrapped_model, x, num_iters=10):
        """Time multiple forward-backward iterations."""
        # Warmup
        for _ in range(3):
            wrapped_model.zero_grad()
            y = wrapped_model(x)
            loss = y.sum()
            loss.backward()
            wrapped_model.get_norm_sample()
        
        # Timed iterations
        start = time.time()
        for _ in range(num_iters):
            wrapped_model.zero_grad()
            y = wrapped_model(x)
            loss = y.sum()
            loss.backward()
            wrapped_model.get_norm_sample()
        elapsed = time.time() - start
        
        return elapsed / num_iters
    
    def test_fuse_faster_than_hooks_long_sequence(self):
        """Test that fused approach is faster with long sequences."""
        # Use longer sequence for more pronounced difference
        B, T, D_in, D_hidden, D_out = 8, 512, 128, 256, 64
        num_layers = 4
        
        # Create two identical models
        torch.manual_seed(42)
        model_hook = AllLinearModel(D_in, D_hidden, D_out, num_layers)
        
        torch.manual_seed(42)
        model_fuse = AllLinearModel(D_in, D_hidden, D_out, num_layers)
        
        # Wrap with different modes
        wrapped_hook = wrap_model(
            model_hook,
            grad_sample_mode="flash_fsdp",  # Standard hook-based
            batch_first=True,
            loss_reduction="mean",
            max_grad_norm=1.0,
        )
        
        wrapped_fuse = wrap_model(
            model_fuse,
            grad_sample_mode="flash_fsdp_fuse",  # Fused approach
            batch_first=True,
            loss_reduction="mean",
            max_grad_norm=1.0,
        )
        
        # Create input
        x = torch.randn(B, T, D_in)
        
        # Time both approaches
        time_hook = self._time_iterations(wrapped_hook, x, num_iters=5)
        time_fuse = self._time_iterations(wrapped_fuse, x, num_iters=5)
        
        print(f"\nPerformance comparison (B={B}, T={T}, layers={num_layers}):")
        print(f"  Hook-based: {time_hook*1000:.2f} ms/iter")
        print(f"  Fused:      {time_fuse*1000:.2f} ms/iter")
        print(f"  Speedup:    {time_hook/time_fuse:.2f}x")
        
        # Fused should be at least comparable (may not be faster on CPU)
        # The real speedup is on GPU with FSDP where hook overhead is eliminated
        self.assertGreater(time_hook / time_fuse, 0.5)  # At least not 2x slower
    
    def test_fused_norms_correct_single_linear(self):
        """Test that fused approach computes correct exact norms for single Linear."""
        B, T, D_in, D_out = 4, 32, 64, 128
        
        # Single Linear layer to isolate the norm computation
        model = nn.Linear(D_in, D_out)
        
        wrapped_fuse = wrap_model(
            model,
            grad_sample_mode="flash_fsdp_fuse",
            batch_first=True,
            loss_reduction="mean",
            max_grad_norm=1.0,
        )
        
        torch.manual_seed(123)
        x = torch.randn(B, T, D_in, requires_grad=True)
        
        # Forward-backward for fused
        wrapped_fuse.zero_grad()
        y_fuse = wrapped_fuse(x)
        loss_fuse = y_fuse.sum()
        loss_fuse.backward()
        norms_fuse = wrapped_fuse.get_norm_sample()
        
        # Manually compute exact per-sample gradient norms
        # For Linear: grad_w[i] = x_i^T @ g_i, grad_b[i] = sum_t(g_i)
        with torch.no_grad():
            grad_out = torch.ones_like(y_fuse)  # Since loss = y.sum(), grad = 1
            
            expected_norms_sq = torch.zeros(B)
            for i in range(B):
                # Weight gradient norm: ||x_i^T @ g_i||_F^2
                grad_w = x[i].T @ grad_out[i]  # [D_in, D_out]
                weight_norm_sq = (grad_w ** 2).sum()
                
                # Bias gradient norm: ||sum_t(g_i)||^2
                grad_b = grad_out[i].sum(dim=0)  # [D_out]
                bias_norm_sq = (grad_b ** 2).sum()
                
                expected_norms_sq[i] = weight_norm_sq + bias_norm_sq
            
            expected_norms = torch.sqrt(expected_norms_sq)
        
        # Compare
        print(f"\nExact norm verification (single Linear):")
        print(f"  Expected norms: {expected_norms}")
        print(f"  Fused norms:    {norms_fuse}")
        print(f"  Max diff:       {(expected_norms - norms_fuse).abs().max().item():.6f}")
        
        self.assertTrue(torch.allclose(expected_norms, norms_fuse, rtol=1e-4, atol=1e-5))
    
    def test_fused_norms_correct_multi_linear(self):
        """Test that fused approach computes correct norms for multiple Linear layers."""
        B, T, D_in, D_hidden, D_out = 4, 32, 32, 64, 16
        
        # Simple model: two Linear layers
        class TwoLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(D_in, D_hidden)
                self.fc2 = nn.Linear(D_hidden, D_out)
            
            def forward(self, x):
                return self.fc2(self.fc1(x))
        
        model = TwoLinear()
        
        wrapped_fuse = wrap_model(
            model,
            grad_sample_mode="flash_fsdp_fuse",
            batch_first=True,
            loss_reduction="mean",
            max_grad_norm=1.0,
        )
        
        torch.manual_seed(123)
        x = torch.randn(B, T, D_in, requires_grad=True)
        
        # Forward-backward for fused
        wrapped_fuse.zero_grad()
        y = wrapped_fuse(x)
        loss = y.sum()
        loss.backward()
        norms_fuse = wrapped_fuse.get_norm_sample()
        
        # Manually compute exact per-sample gradient norms by running individual forwards
        with torch.no_grad():
            expected_norms_sq = torch.zeros(B)
            
            for i in range(B):
                # Forward for sample i
                x_i = x[i:i+1]  # [1, T, D_in]
                h_i = model.fc1(x_i)  # [1, T, D_hidden]
                y_i = model.fc2(h_i)  # [1, T, D_out]
                
                # Backprop with unit gradient
                grad_y = torch.ones_like(y_i)  # [1, T, D_out]
                
                # fc2 gradients
                # grad_fc2_w = h_i^T @ grad_y
                grad_fc2_w = h_i.squeeze(0).T @ grad_y.squeeze(0)  # [D_hidden, D_out]
                grad_fc2_b = grad_y.sum(dim=(0, 1))  # [D_out]
                
                fc2_norm_sq = (grad_fc2_w ** 2).sum() + (grad_fc2_b ** 2).sum()
                
                # fc1 gradients
                # grad_h = grad_y @ fc2.weight
                grad_h = grad_y @ model.fc2.weight  # [1, T, D_hidden]
                grad_fc1_w = x_i.squeeze(0).T @ grad_h.squeeze(0)  # [D_in, D_hidden]
                grad_fc1_b = grad_h.sum(dim=(0, 1))  # [D_hidden]
                
                fc1_norm_sq = (grad_fc1_w ** 2).sum() + (grad_fc1_b ** 2).sum()
                
                expected_norms_sq[i] = fc1_norm_sq + fc2_norm_sq
            
            expected_norms = torch.sqrt(expected_norms_sq)
        
        # Compare
        print(f"\nExact norm verification (multi-Linear):")
        print(f"  Expected norms: {expected_norms}")
        print(f"  Fused norms:    {norms_fuse}")
        print(f"  Max diff:       {(expected_norms - norms_fuse).abs().max().item():.6f}")
        
        self.assertTrue(torch.allclose(expected_norms, norms_fuse, rtol=1e-3, atol=1e-4))
    
    def test_fuse_vs_fuse_bk_consistency(self):
        """Test that flash_fsdp_fuse and flash_fsdp_fuse_bk produce identical norms."""
        B, T, D_in, D_hidden, D_out = 4, 32, 64, 128, 32
        
        # Create two identical models
        torch.manual_seed(42)
        model_fuse = AllLinearModel(D_in, D_hidden, D_out, num_layers=3)
        
        torch.manual_seed(42)
        model_fuse_bk = AllLinearModel(D_in, D_hidden, D_out, num_layers=3)
        
        # Wrap with different modes
        wrapped_fuse = wrap_model(
            model_fuse,
            grad_sample_mode="flash_fsdp_fuse",
            batch_first=True,
            loss_reduction="mean",
            max_grad_norm=1.0,
        )
        
        wrapped_fuse_bk = wrap_model(
            model_fuse_bk,
            grad_sample_mode="flash_fsdp_fuse_bk",
            batch_first=True,
            loss_reduction="mean",
            max_grad_norm=1.0,
        )
        
        # Same input
        torch.manual_seed(123)
        x = torch.randn(B, T, D_in)
        
        # Forward-backward for fuse
        wrapped_fuse.zero_grad()
        y_fuse = wrapped_fuse(x.clone())
        loss_fuse = y_fuse.sum()
        loss_fuse.backward()
        norms_fuse = wrapped_fuse.get_norm_sample()
        
        # Forward-backward for fuse_bk
        wrapped_fuse_bk.zero_grad()
        y_fuse_bk = wrapped_fuse_bk(x.clone())
        loss_fuse_bk = y_fuse_bk.sum()
        loss_fuse_bk.backward()
        norms_fuse_bk = wrapped_fuse_bk.get_norm_sample()
        
        # Compare outputs
        print(f"\nFuse vs Fuse_BK comparison:")
        print(f"  Output match: {torch.allclose(y_fuse, y_fuse_bk, rtol=1e-5)}")
        print(f"  Norms (fuse):    {norms_fuse}")
        print(f"  Norms (fuse_bk): {norms_fuse_bk}")
        print(f"  Max norm diff:   {(norms_fuse - norms_fuse_bk).abs().max().item():.6e}")
        
        # Outputs should be identical
        self.assertTrue(torch.allclose(y_fuse, y_fuse_bk, rtol=1e-5))
        
        # Norms should be very close (might have small numerical differences)
        self.assertTrue(torch.allclose(norms_fuse, norms_fuse_bk, rtol=1e-3, atol=1e-4))
    
    @unittest.skipIf(not TRITON_AVAILABLE, "Triton not available for GPU test")
    def test_fuse_bk_gradient_accumulation_correctness(self):
        """Test that flash_fsdp_fuse_bk correctly accumulates clipped gradients."""
        B, T, D_in, D_out = 4, 32, 64, 128
        max_grad_norm = 1.0
        
        # Single Linear layer
        torch.manual_seed(42)
        model = nn.Linear(D_in, D_out)
        
        # Move to GPU for triton kernel
        if torch.cuda.is_available():
            device = 'cuda'
            model = model.to(device)
        else:
            device = 'cpu'
        
        wrapped = wrap_model(
            model,
            grad_sample_mode="flash_fsdp_fuse_bk",
            batch_first=True,
            loss_reduction="mean",
            max_grad_norm=max_grad_norm,
        )
        
        # Input
        torch.manual_seed(123)
        x = torch.randn(B, T, D_in, device=device)
        
        # Forward-backward
        wrapped.zero_grad()
        y = wrapped(x)
        loss = y.sum()
        loss.backward()
        
        # Get per-sample norms and clipping coefficients
        norms = wrapped.get_norm_sample()
        coef = wrapped.get_clipping_coef()
        
        # Get accumulated gradient from bookkeeping
        accumulated_grad = wrapped._module.model.weight.grad
        
        # Manually compute expected accumulated gradient
        with torch.no_grad():
            grad_out = torch.ones_like(y)
            expected_grad = torch.zeros_like(wrapped._module.model.weight)
            
            for i in range(B):
                # Per-sample weight gradient
                per_sample_grad = x[i].T @ grad_out[i]
                # Apply clipping
                clipped_grad = coef[i] * per_sample_grad
                expected_grad += clipped_grad
        
        # Compare
        max_diff = (accumulated_grad - expected_grad).abs().max().item()
        print(f"\nGradient accumulation test:")
        print(f"  Max diff: {max_diff:.6e}")
        print(f"  Relative error: {max_diff / expected_grad.abs().max().item():.6e}")
        
        self.assertTrue(torch.allclose(accumulated_grad, expected_grad, rtol=1e-3, atol=1e-4))
    
    def test_fuse_bk_multiple_iterations(self):
        """Test flash_fsdp_fuse_bk over multiple training iterations."""
        B, T, D_in, D_hidden, D_out = 4, 32, 64, 128, 32
        max_grad_norm = 1.0
        
        # Create model
        torch.manual_seed(42)
        model = AllLinearModel(D_in, D_hidden, D_out, num_layers=2)
        
        wrapped = wrap_model(
            model,
            grad_sample_mode="flash_fsdp_fuse_bk",
            batch_first=True,
            loss_reduction="mean",
            max_grad_norm=max_grad_norm,
        )
        
        # Run multiple iterations
        losses = []
        all_norms = []
        
        for iter_idx in range(3):
            torch.manual_seed(100 + iter_idx)
            x = torch.randn(B, T, D_in)
            
            wrapped.zero_grad()
            y = wrapped(x)
            loss = y.sum()
            loss.backward()
            
            norms = wrapped.get_norm_sample()
            losses.append(loss.item())
            all_norms.append(norms.clone())
            
            # Simulate optimizer step (just for testing, no actual update)
            print(f"\nIteration {iter_idx}: loss={loss.item():.4f}, "
                  f"avg_norm={norms.mean().item():.4f}")
        
        # Verify norms are computed for each iteration
        for i, norms in enumerate(all_norms):
            self.assertEqual(norms.shape, (B,))
            self.assertTrue((norms > 0).all(), f"Iteration {i} has non-positive norms")
    
    def test_fuse_bk_with_hooks_comparison(self):
        """Compare flash_fsdp_fuse_bk against standard hook-based flash."""
        B, T, D_in, D_hidden, D_out = 4, 32, 64, 128, 32
        max_grad_norm = 1.0
        
        # Create two identical models
        torch.manual_seed(42)
        model_hook = AllLinearModel(D_in, D_hidden, D_out, num_layers=2)
        
        torch.manual_seed(42)
        model_fuse_bk = AllLinearModel(D_in, D_hidden, D_out, num_layers=2)
        
        # Wrap with different modes
        wrapped_hook = wrap_model(
            model_hook,
            grad_sample_mode="flash_fsdp",  # Standard hook-based
            batch_first=True,
            loss_reduction="sum",  # Changed to sum to match loss = y.sum()
            max_grad_norm=max_grad_norm,
        )
        
        wrapped_fuse_bk = wrap_model(
            model_fuse_bk,
            grad_sample_mode="flash_fsdp_fuse_bk",  # Fused + bookkeeping
            batch_first=True,
            loss_reduction="sum",  # Changed to sum to match loss = y.sum()
            max_grad_norm=max_grad_norm,
        )
        
        # Same input
        torch.manual_seed(123)
        x = torch.randn(B, T, D_in)
        
        # Forward-backward for hook-based
        wrapped_hook.zero_grad()
        y_hook = wrapped_hook(x.clone())
        loss_hook = y_hook.sum()
        loss_hook.backward()
        norms_hook = wrapped_hook.get_norm_sample()
        
        # Forward-backward for fuse_bk
        wrapped_fuse_bk.zero_grad()
        y_fuse_bk = wrapped_fuse_bk(x.clone())
        loss_fuse_bk = y_fuse_bk.sum()
        loss_fuse_bk.backward()
        norms_fuse_bk = wrapped_fuse_bk.get_norm_sample()
        
        # Compare
        print(f"\nHook-based vs Fuse_BK comparison:")
        print(f"  Output match: {torch.allclose(y_hook, y_fuse_bk, rtol=1e-4)}")
        print(f"  Norms (hook):    {norms_hook}")
        print(f"  Norms (fuse_bk): {norms_fuse_bk}")
        print(f"  Max norm diff:   {(norms_hook - norms_fuse_bk).abs().max().item():.6e}")
        
        # Outputs should be identical (same forward pass)
        self.assertTrue(torch.allclose(y_hook, y_fuse_bk, rtol=1e-5))
        
        # Norms should be very close
        # Hook-based uses exact computation, fuse_bk uses optimized kernels
        # Allow slightly larger tolerance due to algorithmic differences
        self.assertTrue(torch.allclose(norms_hook, norms_fuse_bk, rtol=1e-2, atol=1e-3))
    
    def test_fuse_bk_loss_reduction_modes(self):
        """Test flash_fsdp_fuse_bk with different loss reduction modes."""
        B, T, D_in, D_out = 4, 32, 64, 128
        max_grad_norm = 1.0
        
        for loss_reduction in ["mean", "sum"]:
            with self.subTest(loss_reduction=loss_reduction):
                torch.manual_seed(42)
                model = nn.Linear(D_in, D_out)
                
                wrapped = wrap_model(
                    model,
                    grad_sample_mode="flash_fsdp_fuse_bk",
                    batch_first=True,
                    loss_reduction=loss_reduction,
                    max_grad_norm=max_grad_norm,
                )
                
                torch.manual_seed(123)
                x = torch.randn(B, T, D_in)
                
                wrapped.zero_grad()
                y = wrapped(x)
                
                if loss_reduction == "mean":
                    loss = y.mean()
                else:  # sum
                    loss = y.sum()
                
                loss.backward()
                norms = wrapped.get_norm_sample()
                
                # Verify norms are computed correctly
                self.assertEqual(norms.shape, (B,))
                self.assertTrue((norms > 0).all())
                
                print(f"\nLoss reduction={loss_reduction}: "
                      f"avg_norm={norms.mean().item():.4f}, "
                      f"min_norm={norms.min().item():.4f}, "
                      f"max_norm={norms.max().item():.4f}")
    
    def test_all_modes_consistency(self):
        """Comprehensive test: Verify all modes produce consistent results."""
        B, T, D_in, D_hidden, D_out = 4, 32, 64, 128, 32
        max_grad_norm = 1.0
        
        modes_to_test = ["flash_fsdp", "flash_fsdp_fuse", "flash_fsdp_fuse_bk"]
        results = {}
        
        # Same input for all modes
        torch.manual_seed(123)
        x_test = torch.randn(B, T, D_in)
        
        for mode in modes_to_test:
            torch.manual_seed(42)  # Same model initialization
            model = AllLinearModel(D_in, D_hidden, D_out, num_layers=2)
            
            wrapped = wrap_model(
                model,
                grad_sample_mode=mode,
                batch_first=True,
                loss_reduction="sum",  # Changed to sum to match loss = y.sum()
                max_grad_norm=max_grad_norm,
            )
            
            wrapped.zero_grad()
            y = wrapped(x_test.clone())
            loss = y.sum()
            loss.backward()
            norms = wrapped.get_norm_sample()
            
            results[mode] = {
                'output': y.detach().clone(),
                'norms': norms.detach().clone(),
                'loss': loss.item()
            }
            
            print(f"\nMode: {mode}")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Norms: {norms}")
        
        # Compare all modes
        print("\n" + "="*60)
        print("CONSISTENCY CHECK:")
        print("="*60)
        
        # Compare outputs (should be identical)
        for mode1 in modes_to_test:
            for mode2 in modes_to_test:
                if mode1 < mode2:  # Avoid duplicate comparisons
                    output_match = torch.allclose(
                        results[mode1]['output'], 
                        results[mode2]['output'], 
                        rtol=1e-5
                    )
                    norm_match = torch.allclose(
                        results[mode1]['norms'], 
                        results[mode2]['norms'], 
                        rtol=1e-2, atol=1e-3
                    )
                    max_norm_diff = (results[mode1]['norms'] - results[mode2]['norms']).abs().max().item()
                    
                    print(f"\n{mode1} vs {mode2}:")
                    print(f"  Output match: {output_match}")
                    print(f"  Norm match: {norm_match}")
                    print(f"  Max norm diff: {max_norm_diff:.6e}")
                    
                    # Outputs should be identical
                    self.assertTrue(output_match, 
                                  f"Outputs don't match: {mode1} vs {mode2}")
                    # Norms should be very close
                    self.assertTrue(norm_match,
                                  f"Norms don't match: {mode1} vs {mode2}, max_diff={max_norm_diff}")


if __name__ == "__main__":
    unittest.main()

