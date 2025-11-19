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
Overall correctness test for convolutional flash clipping with DP training.

This test verifies that:
1. Flash clipping + bookkeeping produces similar results to non-DP training
2. With very small noise and large clipping threshold, DP training matches non-DP
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from opacus.grad_sample import GradSampleModuleFastGradientClipping
from opacus.optimizers import DPOptimizerFastGradientClipping
from opacus.utils.fast_gradient_clipping_utils import DPLossFastGradientClipping
from opacus.utils.per_sample_gradients_utils import clone_module


class SimpleConvNet(nn.Module):
    """Simple ConvNet for testing flash clipping with DP."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ConvFlashClippingCorrectnessTest(unittest.TestCase):
    """Test overall correctness of flash clipping for convolutional layers with DP."""
    
    def setUp(self):
        self.batch_size = 8
        self.num_batches = 3
        self.num_classes = 10
        self.learning_rate = 0.01
        
        # DP parameters - very small noise and large clipping for similarity to non-DP
        self.noise_multiplier = 0.01
        self.max_grad_norm = 100.0
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create synthetic dataset
        total_samples = self.batch_size * self.num_batches
        self.images = torch.randn(total_samples, 3, 32, 32)
        self.labels = torch.randint(0, self.num_classes, (total_samples,))
        
        dataset = TensorDataset(self.images, self.labels)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
    
    def _train_model(self, model, use_dp=False, use_flash_clipping=False, enable_bookkeeping=False):
        """Train a model with or without DP."""
        model.train()
        
        if use_dp:
            # Setup DP training with flash clipping and bookkeeping
            gsm = GradSampleModuleFastGradientClipping(
                model,
                batch_first=True,
                loss_reduction="mean",
                max_grad_norm=self.max_grad_norm,
                use_ghost_clipping=True,
                use_flash_clipping=use_flash_clipping,
                enable_fastdp_bookkeeping=enable_bookkeeping,
            )
            
            optimizer = torch.optim.SGD(gsm.parameters(), lr=self.learning_rate)
            dp_optimizer = DPOptimizerFastGradientClipping(
                optimizer,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
                expected_batch_size=self.batch_size,
                loss_reduction="mean",
            )
            
            criterion_base = nn.CrossEntropyLoss(reduction="mean")
            criterion = DPLossFastGradientClipping(
                gsm, dp_optimizer, criterion_base, loss_reduction="mean"
            )
            
            model_to_train = gsm
            optimizer_to_use = dp_optimizer
        else:
            # Non-DP training
            optimizer_to_use = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
            criterion = nn.CrossEntropyLoss(reduction="mean")
            model_to_train = model
        
        losses = []
        for data, target in self.dataloader:
            optimizer_to_use.zero_grad()
            output = model_to_train(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer_to_use.step()
            losses.append(loss.item())
        
        return losses
    
    def test_flash_clipping_bookkeeping_vs_non_dp(self):
        """
        Test that flash clipping + bookkeeping produces similar results to non-DP.
        
        With very small noise (0.01) and large clipping threshold (100.0), DP training
        should produce results very close to non-DP training.
        """
        # Create two identical models
        base_model = SimpleConvNet(num_classes=self.num_classes)
        
        non_dp_model = clone_module(base_model)
        dp_model = clone_module(base_model)
        
        # Train non-DP model
        non_dp_losses = self._train_model(non_dp_model, use_dp=False)
        
        # Train DP model with flash clipping and bookkeeping
        dp_losses = self._train_model(
            dp_model, 
            use_dp=True, 
            use_flash_clipping=True,
            enable_bookkeeping=True
        )
        
        # Losses should be similar (not identical due to noise)
        # With very small noise multiplier, they should be quite close
        for i, (non_dp_loss, dp_loss) in enumerate(zip(non_dp_losses, dp_losses)):
            relative_diff = abs(non_dp_loss - dp_loss) / (abs(non_dp_loss) + 1e-8)
            self.assertLess(
                relative_diff, 0.2,  # Allow 20% relative difference
                msg=f"Batch {i}: Loss difference too large. Non-DP: {non_dp_loss:.4f}, DP: {dp_loss:.4f}"
            )
        
        # Compare final model parameters
        non_dp_params = torch.cat([p.detach().flatten() for p in non_dp_model.parameters()])
        dp_params = torch.cat([p.detach().flatten() for p in dp_model.parameters()])
        
        # Parameters should be similar
        param_diff = torch.norm(non_dp_params - dp_params) / torch.norm(non_dp_params)
        self.assertLess(
            param_diff.item(), 0.15,  # Allow 15% relative difference in parameters
            msg=f"Model parameters diverged too much: {param_diff.item():.4f}"
        )
        
        print(f"✓ Flash clipping + bookkeeping test passed")
        print(f"  Final loss - Non-DP: {non_dp_losses[-1]:.4f}, DP: {dp_losses[-1]:.4f}")
        print(f"  Parameter relative difference: {param_diff.item():.4f}")
    
    def test_flash_clipping_norm_computation(self):
        """Test that flash clipping correctly computes norms for Conv2d layers."""
        from opacus.grad_sample.conv import (
            compute_conv_grad_sample,
            compute_conv_norm_sample_flash,
        )
        
        # Create a simple Conv2d layer
        layer = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True)
        layer.eval()
        
        # Create test data
        batch_size = 4
        activations = torch.randn(batch_size, 3, 8, 8)
        output = layer(activations)
        backprops = torch.randn_like(output)
        
        # Compute norms using flash clipping
        flash_norms = compute_conv_norm_sample_flash(layer, [activations], backprops)
        
        # Compute expected norms from full per-sample gradients
        grad_samples = compute_conv_grad_sample(layer, [activations], backprops)
        expected_norms = {}
        for param, gs in grad_samples.items():
            gs_flat = gs.reshape(gs.shape[0], -1)
            expected_norms[param] = torch.norm(gs_flat, dim=1, p=2)
        
        # Compare
        for param in expected_norms.keys():
            torch.testing.assert_close(
                flash_norms[param],
                expected_norms[param],
                rtol=1e-4,
                atol=1e-5,
                msg=f"Flash clipping norm mismatch for {param}",
            )
        
        print(f"✓ Flash clipping norm computation test passed")
        print(f"  Weight norms (first 3): {flash_norms[layer.weight][:3].tolist()}")
        if layer.bias is not None and layer.bias.requires_grad:
            print(f"  Bias norms (first 3): {flash_norms[layer.bias][:3].tolist()}")
    
    def test_ghost_vs_flash_clipping_training(self):
        """Test that ghost and flash clipping produce similar training results."""
        # Create two identical models
        base_model = SimpleConvNet(num_classes=self.num_classes)
        
        ghost_model = clone_module(base_model)
        flash_model = clone_module(base_model)
        
        # Train with ghost clipping (use_flash_clipping=False)
        ghost_losses = self._train_model(
            ghost_model,
            use_dp=True,
            use_flash_clipping=False,
            enable_bookkeeping=False
        )
        
        # Train with flash clipping (use_flash_clipping=True)
        flash_losses = self._train_model(
            flash_model,
            use_dp=True,
            use_flash_clipping=True,
            enable_bookkeeping=False
        )
        
        # Losses should be very similar (both use same noise)
        for i, (ghost_loss, flash_loss) in enumerate(zip(ghost_losses, flash_losses)):
            # Allow for small numerical differences
            relative_diff = abs(ghost_loss - flash_loss) / (abs(ghost_loss) + 1e-8)
            self.assertLess(
                relative_diff, 0.05,  # Allow 5% relative difference
                msg=f"Batch {i}: Ghost vs Flash loss difference too large. Ghost: {ghost_loss:.4f}, Flash: {flash_loss:.4f}"
            )
        
        print(f"✓ Ghost vs Flash clipping training test passed")
        print(f"  Final loss - Ghost: {ghost_losses[-1]:.4f}, Flash: {flash_losses[-1]:.4f}")


if __name__ == "__main__":
    unittest.main()

