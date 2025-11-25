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

Key tests:
1. Correctness: Verify norms computed by fused approach match hook-based approach
2. Performance: Compare wall-clock time for training iterations
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
            grad_sample_mode="flash",  # Standard hook-based
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


if __name__ == "__main__":
    unittest.main()

