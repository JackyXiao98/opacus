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
Tests for convolutional layer norm samplers used in Ghost Clipping and Flash Clipping.

These tests verify that:
1. Standard norm samplers compute correct per-sample gradient norms
2. Flash norm samplers match standard norm samplers
3. Both handle various configurations (kernel sizes, strides, padding, groups, bias)
"""

import unittest
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from opacus.grad_sample.conv import (
    compute_conv_grad_sample,
    compute_conv_norm_sample,
    compute_conv_norm_sample_flash,
    compute_conv_norm_sample_flash_wrapper,
)


class ConvNormSampleTest(unittest.TestCase):
    """Test suite for convolutional layer norm samplers."""

    def _compute_expected_norm_from_grad_sample(
        self,
        layer: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d],
        activations: torch.Tensor,
        backprops: torch.Tensor,
    ) -> dict:
        """
        Compute expected per-sample gradient norms by:
        1. Computing full per-sample gradients
        2. Computing norm of each per-sample gradient
        
        This serves as ground truth for testing norm samplers.
        """
        # Compute per-sample gradients
        grad_samples = compute_conv_grad_sample(layer, [activations], backprops)
        
        # Compute norms from grad samples
        norms = {}
        for param, gs in grad_samples.items():
            # gs shape: [batch_size, ...param_shape]
            # Flatten all dimensions except batch
            gs_flat = gs.reshape(gs.shape[0], -1)
            # Compute norm for each sample
            norms[param] = torch.norm(gs_flat, dim=1, p=2)
        
        return norms

    def _test_conv_norm_sample_correctness(
        self,
        layer: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d],
        input_shape: tuple,
        rtol: float = 1e-4,
        atol: float = 1e-6,
    ):
        """
        Test that norm sampler produces correct norms compared to ground truth.
        
        Args:
            layer: Conv layer to test
            input_shape: Shape of input tensor (batch, channels, *spatial)
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison
        """
        layer.eval()  # Disable dropout if any
        
        # Create random inputs
        batch_size = input_shape[0]
        activations = torch.randn(input_shape, requires_grad=True)
        
        # Forward pass
        output = layer(activations)
        
        # Create random backprops (same shape as output)
        backprops = torch.randn_like(output)
        
        # Compute expected norms from full gradient computation
        expected_norms = self._compute_expected_norm_from_grad_sample(
            layer, activations, backprops
        )
        
        # Compute norms using norm sampler
        computed_norms = compute_conv_norm_sample(layer, [activations], backprops)
        
        # Compare norms for each parameter
        for param in expected_norms.keys():
            expected = expected_norms[param]
            computed = computed_norms[param]
            
            # Check shapes match
            self.assertEqual(
                expected.shape,
                computed.shape,
                f"Norm shape mismatch for {param}: expected {expected.shape}, got {computed.shape}",
            )
            
            # Check values are close
            torch.testing.assert_close(
                computed,
                expected,
                rtol=rtol,
                atol=atol,
                msg=f"Norm values mismatch for {param}",
            )

    def test_conv2d_basic(self):
        """Test Conv2d with basic configuration."""
        layer = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        input_shape = (4, 3, 8, 8)  # batch=4, channels=3, height=8, width=8
        self._test_conv_norm_sample_correctness(layer, input_shape)

    def test_conv2d_no_bias(self):
        """Test Conv2d without bias."""
        layer = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        input_shape = (4, 3, 8, 8)
        self._test_conv_norm_sample_correctness(layer, input_shape)

    def test_conv2d_different_kernel_sizes(self):
        """Test Conv2d with various kernel sizes."""
        for kernel_size in [1, 3, 5]:
            with self.subTest(kernel_size=kernel_size):
                layer = nn.Conv2d(
                    in_channels=3,
                    out_channels=8,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=True,
                )
                input_shape = (2, 3, 8, 8)
                self._test_conv_norm_sample_correctness(layer, input_shape)

    def test_conv2d_different_strides(self):
        """Test Conv2d with different strides."""
        for stride in [1, 2]:
            with self.subTest(stride=stride):
                layer = nn.Conv2d(
                    in_channels=3,
                    out_channels=8,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=True,
                )
                input_shape = (2, 3, 8, 8)
                self._test_conv_norm_sample_correctness(layer, input_shape)

    def test_conv2d_different_padding(self):
        """Test Conv2d with different padding."""
        for padding in [0, 1, 2]:
            with self.subTest(padding=padding):
                layer = nn.Conv2d(
                    in_channels=3,
                    out_channels=8,
                    kernel_size=3,
                    stride=1,
                    padding=padding,
                    bias=True,
                )
                input_shape = (2, 3, 8, 8)
                self._test_conv_norm_sample_correctness(layer, input_shape)

    def test_conv2d_grouped(self):
        """Test grouped Conv2d."""
        layer = nn.Conv2d(
            in_channels=4,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=2,
            bias=True,
        )
        input_shape = (2, 4, 8, 8)
        self._test_conv_norm_sample_correctness(layer, input_shape)

    def test_conv2d_depthwise(self):
        """Test depthwise separable Conv2d (groups=in_channels=out_channels)."""
        channels = 8
        layer = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,
            bias=True,
        )
        input_shape = (2, channels, 8, 8)
        self._test_conv_norm_sample_correctness(layer, input_shape)

    def test_conv2d_single_sample(self):
        """Test Conv2d with batch size of 1."""
        layer = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        input_shape = (1, 3, 8, 8)
        self._test_conv_norm_sample_correctness(layer, input_shape)

    def test_conv2d_large_batch(self):
        """Test Conv2d with larger batch size."""
        layer = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        input_shape = (16, 3, 8, 8)
        self._test_conv_norm_sample_correctness(layer, input_shape)

    def test_conv1d_basic(self):
        """Test Conv1d with basic configuration."""
        layer = nn.Conv1d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        input_shape = (4, 3, 16)  # batch=4, channels=3, length=16
        self._test_conv_norm_sample_correctness(layer, input_shape)

    def test_conv1d_no_bias(self):
        """Test Conv1d without bias."""
        layer = nn.Conv1d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        input_shape = (4, 3, 16)
        self._test_conv_norm_sample_correctness(layer, input_shape)

    def test_conv3d_basic(self):
        """Test Conv3d with basic configuration."""
        layer = nn.Conv3d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        input_shape = (2, 3, 4, 4, 4)  # batch=2, channels=3, D=4, H=4, W=4
        self._test_conv_norm_sample_correctness(layer, input_shape)

    def test_conv3d_no_bias(self):
        """Test Conv3d without bias."""
        layer = nn.Conv3d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        input_shape = (2, 3, 4, 4, 4)
        self._test_conv_norm_sample_correctness(layer, input_shape)

    def test_flash_norm_sampler_matches_standard(self):
        """Test that flash norm sampler produces same results as standard."""
        layer = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        layer.eval()
        
        input_shape = (4, 3, 8, 8)
        activations = torch.randn(input_shape)
        output = layer(activations)
        backprops = torch.randn_like(output)
        
        # Compute with standard norm sampler
        standard_norms = compute_conv_norm_sample(layer, [activations], backprops)
        
        # Compute with flash norm sampler
        flash_norms = compute_conv_norm_sample_flash_wrapper(layer, [activations], backprops)
        
        # Compare
        for param in standard_norms.keys():
            torch.testing.assert_close(
                flash_norms[param],
                standard_norms[param],
                rtol=1e-5,
                atol=1e-7,
                msg=f"Flash and standard norm samplers disagree for {param}",
            )

    def test_empty_batch(self):
        """Test handling of empty batch."""
        layer = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        
        # Empty batch
        input_shape = (0, 3, 8, 8)
        activations = torch.randn(input_shape)
        output = layer(activations)
        backprops = torch.randn_like(output)
        
        norms = compute_conv_norm_sample(layer, [activations], backprops)
        
        # Check that norms are empty tensors with correct shape
        for param, norm in norms.items():
            self.assertEqual(norm.shape[0], 0, f"Expected empty norm for {param}")

    def test_numerical_stability(self):
        """Test numerical stability with very small and very large values."""
        layer = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        layer.eval()
        
        # Test with very small values
        input_shape = (4, 3, 8, 8)
        activations = torch.randn(input_shape) * 1e-6
        output = layer(activations)
        backprops = torch.randn_like(output) * 1e-6
        
        norms_small = compute_conv_norm_sample(layer, [activations], backprops)
        
        # Check norms are non-negative and finite
        for param, norm in norms_small.items():
            self.assertTrue(torch.all(norm >= 0), f"Negative norm found for {param}")
            self.assertTrue(torch.all(torch.isfinite(norm)), f"Non-finite norm found for {param}")
        
        # Test with larger values
        activations = torch.randn(input_shape) * 10
        output = layer(activations)
        backprops = torch.randn_like(output) * 10
        
        norms_large = compute_conv_norm_sample(layer, [activations], backprops)
        
        for param, norm in norms_large.items():
            self.assertTrue(torch.all(norm >= 0), f"Negative norm found for {param}")
            self.assertTrue(torch.all(torch.isfinite(norm)), f"Non-finite norm found for {param}")


class FlashClippingAccuracyTest(unittest.TestCase):
    """
    Test flash clipping accuracy against ghost clipping and ground truth.
    
    Flash clipping should produce identical results to ghost clipping,
    just more efficiently by avoiding materialization of large (spatial, spatial) matrices.
    """
    
    def _compute_expected_norm_from_grad_sample(
        self,
        layer: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d],
        activations: torch.Tensor,
        backprops: torch.Tensor,
    ) -> dict:
        """Compute ground truth norms from full per-sample gradients."""
        grad_samples = compute_conv_grad_sample(layer, [activations], backprops)
        
        norms = {}
        for param, gs in grad_samples.items():
            gs_flat = gs.reshape(gs.shape[0], -1)
            norms[param] = torch.norm(gs_flat, dim=1, p=2)
        
        return norms
    
    def test_flash_vs_ghost_clipping_equivalence(self):
        """Test that flash clipping produces same results as ghost clipping."""
        configs = [
            # (in_channels, out_channels, kernel_size, stride, padding, groups, bias)
            (3, 16, 3, 1, 1, 1, True),    # Basic
            (3, 16, 3, 1, 1, 1, False),   # No bias
            (3, 16, 5, 1, 2, 1, True),    # Larger kernel
            (3, 16, 3, 2, 1, 1, True),    # Stride 2
            (3, 16, 3, 1, 0, 1, True),    # No padding
            (4, 8, 3, 1, 1, 2, True),     # Grouped (groups=2)
            (8, 8, 3, 1, 1, 8, True),     # Depthwise
        ]
        
        for in_ch, out_ch, k_size, stride, padding, groups, bias in configs:
            with self.subTest(
                in_ch=in_ch, out_ch=out_ch, k=k_size, 
                stride=stride, pad=padding, groups=groups, bias=bias
            ):
                layer = nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=k_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=bias,
                )
                layer.eval()
                
                # Create test data
                input_shape = (4, in_ch, 16, 16)
                activations = torch.randn(input_shape)
                output = layer(activations)
                backprops = torch.randn_like(output)
                
                # Compute with ghost clipping
                ghost_norms = compute_conv_norm_sample(layer, [activations], backprops)
                
                # Compute with flash clipping
                flash_norms = compute_conv_norm_sample_flash(layer, [activations], backprops)
                
                # Compare
                for param in ghost_norms.keys():
                    torch.testing.assert_close(
                        flash_norms[param],
                        ghost_norms[param],
                        rtol=1e-4,
                        atol=1e-5,
                        msg=f"Flash and ghost clipping disagree for {param}",
                    )
    
    def test_flash_clipping_conv2d_various_configs(self):
        """Test flash clipping accuracy for Conv2d with various configurations."""
        configs = [
            # (in_channels, out_channels, kernel_size, stride, padding, groups, bias)
            (3, 16, 3, 1, 1, 1, True),    # Basic
            (3, 16, 3, 1, 1, 1, False),   # No bias
            (3, 16, 5, 1, 2, 1, True),    # Larger kernel
            (3, 16, 3, 2, 1, 1, True),    # Stride 2
            (3, 16, 3, 1, 0, 1, True),    # No padding
            (4, 8, 3, 1, 1, 2, True),     # Grouped (groups=2)
            (8, 8, 3, 1, 1, 8, True),     # Depthwise
        ]
        
        for in_ch, out_ch, k_size, stride, padding, groups, bias in configs:
            with self.subTest(
                in_ch=in_ch, out_ch=out_ch, k=k_size, 
                stride=stride, pad=padding, groups=groups, bias=bias
            ):
                layer = nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=k_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=bias,
                )
                layer.eval()
                
                # Create test data
                input_shape = (4, in_ch, 16, 16)
                activations = torch.randn(input_shape)
                output = layer(activations)
                backprops = torch.randn_like(output)
                
                # Compute expected norms from full gradients
                expected_norms = self._compute_expected_norm_from_grad_sample(
                    layer, activations, backprops
                )
                
                # Compute with flash clipping
                flash_norms = compute_conv_norm_sample_flash(layer, [activations], backprops)
                
                # Compare
                for param in expected_norms.keys():
                    torch.testing.assert_close(
                        flash_norms[param],
                        expected_norms[param],
                        rtol=1e-4,
                        atol=1e-5,
                        msg=f"Flash clipping norm mismatch for {param}",
                    )
    
    def test_flash_clipping_conv1d_accuracy(self):
        """Test flash clipping accuracy for Conv1d."""
        layer = nn.Conv1d(
            in_channels=4,
            out_channels=8,
            kernel_size=5,
            stride=2,
            padding=2,
            bias=True,
        )
        layer.eval()
        
        input_shape = (8, 4, 32)  # (batch, channels, length)
        activations = torch.randn(input_shape)
        output = layer(activations)
        backprops = torch.randn_like(output)
        
        expected_norms = self._compute_expected_norm_from_grad_sample(
            layer, activations, backprops
        )
        flash_norms = compute_conv_norm_sample(layer, [activations], backprops)
        
        for param in expected_norms.keys():
            torch.testing.assert_close(
                flash_norms[param],
                expected_norms[param],
                rtol=1e-4,
                atol=1e-5,
                msg=f"Flash clipping norm mismatch for Conv1d {param}",
            )
    
    def test_flash_clipping_conv3d_accuracy(self):
        """Test flash clipping accuracy for Conv3d."""
        layer = nn.Conv3d(
            in_channels=2,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        layer.eval()
        
        input_shape = (4, 2, 8, 8, 8)  # (batch, channels, D, H, W)
        activations = torch.randn(input_shape)
        output = layer(activations)
        backprops = torch.randn_like(output)
        
        expected_norms = self._compute_expected_norm_from_grad_sample(
            layer, activations, backprops
        )
        flash_norms = compute_conv_norm_sample_flash(layer, [activations], backprops)
        
        for param in expected_norms.keys():
            torch.testing.assert_close(
                flash_norms[param],
                expected_norms[param],
                rtol=1e-4,
                atol=1e-5,
                msg=f"Flash clipping norm mismatch for Conv3d {param}",
            )
    
    def test_flash_clipping_large_spatial_dimensions(self):
        """
        Test flash clipping with large spatial dimensions.
        
        This is where flash clipping really shines - avoiding (q, q) matrices
        where q is the spatial dimension product can save significant memory.
        For 32x32 output, ghost clipping would create 1024x1024 matrices.
        """
        layer = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        layer.eval()
        
        # Large spatial dimensions
        input_shape = (4, 3, 32, 32)
        activations = torch.randn(input_shape)
        output = layer(activations)
        backprops = torch.randn_like(output)
        
        expected_norms = self._compute_expected_norm_from_grad_sample(
            layer, activations, backprops
        )
        flash_norms = compute_conv_norm_sample_flash(layer, [activations], backprops)
        
        for param in expected_norms.keys():
            torch.testing.assert_close(
                flash_norms[param],
                expected_norms[param],
                rtol=1e-4,
                atol=1e-5,
                msg=f"Flash clipping norm mismatch for large spatial dims {param}",
            )
    
    def test_flash_clipping_grouped_conv_accuracy(self):
        """Test flash clipping with various group configurations."""
        test_cases = [
            # (in_channels, out_channels, groups)
            (4, 8, 2),   # Standard grouped
            (6, 12, 3),  # 3 groups
            (8, 8, 4),   # 4 groups
            (16, 16, 16),  # Depthwise
        ]
        
        for in_ch, out_ch, groups in test_cases:
            with self.subTest(in_ch=in_ch, out_ch=out_ch, groups=groups):
                layer = nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=groups,
                    bias=True,
                )
                layer.eval()
                
                input_shape = (4, in_ch, 8, 8)
                activations = torch.randn(input_shape)
                output = layer(activations)
                backprops = torch.randn_like(output)
                
                expected_norms = self._compute_expected_norm_from_grad_sample(
                    layer, activations, backprops
                )
                flash_norms = compute_conv_norm_sample_flash(layer, [activations], backprops)
                
                for param in expected_norms.keys():
                    torch.testing.assert_close(
                        flash_norms[param],
                        expected_norms[param],
                        rtol=1e-4,
                        atol=1e-5,
                        msg=f"Flash clipping norm mismatch for grouped conv {param}",
                    )


if __name__ == "__main__":
    unittest.main()

