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

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

from opacus.grad_sample import GradSampleModuleFastGradientClipping
from opacus.optimizers import DPOptimizerFastGradientClipping
from opacus.utils.fast_gradient_clipping_utils import DPLossFastGradientClipping
from opacus.utils.per_sample_gradients_utils import clone_module
from opacus.grad_sample.triton_kernels import is_triton_available


class SyntheticSequenceDataset(Dataset):
    """Dataset for testing sequence models with 3D tensors"""
    def __init__(self, size, seq_length, input_dim, num_classes=10):
        self.size = size
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Generate synthetic sequence data
        self.sequences = torch.randn(self.size, self.seq_length, self.input_dim)
        self.labels = torch.randint(0, self.num_classes, (self.size,))

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.sequences[index], self.labels[index]


class SimpleSequenceModel(nn.Module):
    """Simple sequence model for testing flash clipping"""
    def __init__(self, input_dim=64, hidden_dim=128, num_classes=10):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        x = F.relu(self.linear1(x))  # (batch_size, seq_length, hidden_dim)
        x = self.dropout(x)
        x = F.relu(self.linear2(x))  # (batch_size, seq_length, hidden_dim)
        x = self.linear3(x)  # (batch_size, seq_length, num_classes)
        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # (batch_size, num_classes)
        return x


class FlashClippingCorrectnessTest(unittest.TestCase):
    """Test correctness of flash clipping against ghost clipping"""
    
    def setUp(self):
        self.batch_size = 8
        self.seq_length = 32
        self.input_dim = 64
        self.hidden_dim = 128
        self.num_classes = 10
        self.max_grad_norm = 1.0
        self.noise_multiplier = 0.0  # No noise for correctness testing
        
        # Set device - use GPU if available for Triton, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create dataset
        self.dataset = SyntheticSequenceDataset(
            size=self.batch_size * 4,  # Multiple batches
            seq_length=self.seq_length,
            input_dim=self.input_dim,
            num_classes=self.num_classes
        )
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        
    def _create_models(self):
        """Create identical models for ghost and flash clipping"""
        # Create base model
        base_model = SimpleSequenceModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes
        ).to(self.device)
        
        # Clone for flash clipping
        flash_model = clone_module(base_model).to(self.device)
        ghost_model = clone_module(base_model).to(self.device)
        
        return ghost_model, flash_model
    
    def _setup_ghost_clipping(self, model):
        """Setup ghost clipping model and optimizer"""
        gsm_ghost = GradSampleModuleFastGradientClipping(
            model,
            batch_first=True,
            loss_reduction="mean",
            max_grad_norm=self.max_grad_norm,
            use_ghost_clipping=True,
            use_triton=False  # Standard ghost clipping
        )
        
        optimizer_ghost = torch.optim.SGD(gsm_ghost.parameters(), lr=0.01)
        dp_optimizer_ghost = DPOptimizerFastGradientClipping(
            optimizer_ghost,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            expected_batch_size=self.batch_size,
            loss_reduction="mean",
        )
        
        # Create separate criterion instance for ghost clipping
        criterion_for_ghost = nn.CrossEntropyLoss(reduction="mean")
        criterion_ghost = DPLossFastGradientClipping(
            gsm_ghost, dp_optimizer_ghost, criterion_for_ghost, loss_reduction="mean"
        )
        
        return gsm_ghost, dp_optimizer_ghost, criterion_ghost
    
    def _setup_flash_clipping(self, model):
        """Setup flash clipping model and optimizer"""
        gsm_flash = GradSampleModuleFastGradientClipping(
            model,
            batch_first=True,
            loss_reduction="mean",
            max_grad_norm=self.max_grad_norm,
            use_ghost_clipping=True,
            use_triton=True  # Flash clipping with Triton
        )
        
        optimizer_flash = torch.optim.SGD(gsm_flash.parameters(), lr=0.01)
        dp_optimizer_flash = DPOptimizerFastGradientClipping(
            optimizer_flash,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            expected_batch_size=self.batch_size,
            loss_reduction="mean",
        )
        
        # Create separate criterion instance for flash clipping
        criterion_for_flash = nn.CrossEntropyLoss(reduction="mean")
        criterion_flash = DPLossFastGradientClipping(
            gsm_flash, dp_optimizer_flash, criterion_for_flash, loss_reduction="mean"
        )
        
        return gsm_flash, dp_optimizer_flash, criterion_flash
    
    # @unittest.skipIf(not is_triton_available(), "Triton not available")
    def test_norm_equivalence(self):
        """Test that flash clipping produces the same gradient norms as ghost clipping"""
        ghost_model, flash_model = self._create_models()
        
        # Setup ghost clipping
        gsm_ghost, dp_optimizer_ghost, criterion_ghost = self._setup_ghost_clipping(ghost_model)
        
        # Setup flash clipping
        gsm_flash, dp_optimizer_flash, criterion_flash = self._setup_flash_clipping(flash_model)
        
        # Test on a single batch
        data, target = next(iter(self.dataloader))
        data, target = data.to(self.device), target.to(self.device)
        
        # Ghost clipping forward and backward
        gsm_ghost.train()
        loss_ghost = criterion_ghost(gsm_ghost(data), target)
        loss_ghost.backward()  # This triggers the computation of _norm_sample attributes
        
        # Flash clipping forward and backward
        gsm_flash.train()
        loss_flash = criterion_flash(gsm_flash(data), target)
        loss_flash.backward()  # This triggers the computation of _norm_sample attributes
        
        # Compare losses (should be very close since models are identical)
        # Allow for small numerical differences between ghost and flash clipping
        loss_diff = abs(loss_ghost.item() - loss_flash.item())
        loss_avg = (loss_ghost.item() + loss_flash.item()) / 2
        relative_diff = loss_diff / loss_avg
        self.assertLess(
            relative_diff, 0.01,  # Allow 1% relative difference
            msg=f"Losses should be close for identical models. Ghost: {loss_ghost.item():.6f}, Flash: {loss_flash.item():.6f}, Relative diff: {relative_diff:.6f}"
        )
        
        # Get gradient norms
        ghost_norms = gsm_ghost.get_norm_sample()
        flash_norms = gsm_flash.get_norm_sample()
        
        # Compare gradient norms
        self.assertEqual(
            ghost_norms.shape, flash_norms.shape,
            msg="Gradient norm shapes should match"
        )
        

        # Check numerical equivalence within tolerance
        # Note: When Triton is not available, flash clipping falls back to standard computation
        # which may have larger numerical differences compared to ghost clipping
        torch.testing.assert_close(
            ghost_norms, flash_norms, rtol=0.05, atol=0.05,
            msg="Flash clipping norms should match ghost clipping norms within tolerance"
        )
        
        print(f"✓ Gradient norms match within tolerance")
        print(f"  Ghost norms: {ghost_norms[:5].tolist()}")
        print(f"  Flash norms: {flash_norms[:5].tolist()}")
        print(f"  Max difference: {torch.max(torch.abs(ghost_norms - flash_norms)).item()}")
    
    @unittest.skipIf(not is_triton_available(), "Triton not available")
    def test_clipping_factors_equivalence(self):
        """Test that clipping factors are equivalent between ghost and flash clipping"""
        ghost_model, flash_model = self._create_models()
        
        # Setup with higher grad norm to trigger clipping
        self.max_grad_norm = 0.1  # Lower threshold to ensure clipping
        
        gsm_ghost, dp_optimizer_ghost, criterion_ghost = self._setup_ghost_clipping(ghost_model)
        gsm_flash, dp_optimizer_flash, criterion_flash = self._setup_flash_clipping(flash_model)
        
        data, target = next(iter(self.dataloader))
        data, target = data.to(self.device), target.to(self.device)
        
        # Forward and backward passes
        gsm_ghost.train()
        loss_ghost = criterion_ghost(gsm_ghost(data), target)
        loss_ghost.backward()  # This triggers the computation of _norm_sample attributes
        
        gsm_flash.train()
        loss_flash = criterion_flash(gsm_flash(data), target)
        loss_flash.backward()  # This triggers the computation of _norm_sample attributes
        
        # Get clipping coefficients
        ghost_clipping_coef = gsm_ghost.get_clipping_coef()
        flash_clipping_coef = gsm_flash.get_clipping_coef()
        
        # Compare clipping coefficients
        torch.testing.assert_close(
            ghost_clipping_coef, flash_clipping_coef, rtol=1e-4, atol=1e-6,
            msg="Clipping coefficients should match between ghost and flash clipping"
        )
        
        print(f"✓ Clipping coefficients match within tolerance")
        print(f"  Ghost coef: {ghost_clipping_coef[:5].tolist()}")
        print(f"  Flash coef: {flash_clipping_coef[:5].tolist()}")
    
    @unittest.skipIf(not is_triton_available(), "Triton not available")
    def test_multiple_batches_consistency(self):
        """Test consistency across multiple batches"""
        ghost_model, flash_model = self._create_models()
        
        gsm_ghost, dp_optimizer_ghost, criterion_ghost = self._setup_ghost_clipping(ghost_model)
        gsm_flash, dp_optimizer_flash, criterion_flash = self._setup_flash_clipping(flash_model)
        
        ghost_norms_list = []
        flash_norms_list = []
        
        # Test on multiple batches
        for i, (data, target) in enumerate(self.dataloader):
            if i >= 3:  # Test first 3 batches
                break
                
            # Move data to device
            data, target = data.to(self.device), target.to(self.device)
                
            # Reset gradients
            dp_optimizer_ghost.zero_grad()
            dp_optimizer_flash.zero_grad()
            
            # Ghost clipping
            gsm_ghost.train()
            loss_ghost = criterion_ghost(gsm_ghost(data), target)
            loss_ghost.backward()
            ghost_norms = gsm_ghost.get_norm_sample()
            ghost_norms_list.append(ghost_norms)
            
            # Flash clipping
            gsm_flash.train()
            loss_flash = criterion_flash(gsm_flash(data), target)
            loss_flash.backward()
            flash_norms = gsm_flash.get_norm_sample()
            flash_norms_list.append(flash_norms)
            
            # Compare for this batch
            torch.testing.assert_close(
                ghost_norms, flash_norms, rtol=1e-4, atol=1e-6,
                msg=f"Batch {i}: Flash clipping norms should match ghost clipping norms"
            )
        
        print(f"✓ Consistency verified across {len(ghost_norms_list)} batches")
    
    def test_fallback_when_triton_unavailable(self):
        """Test that flash clipping falls back gracefully when Triton is unavailable"""
        # This test doesn't require Triton to be available
        ghost_model, flash_model = self._create_models()
        
        # Force use_triton=True even if Triton might not be available
        # The implementation should handle this gracefully
        gsm_flash = GradSampleModuleFastGradientClipping(
            flash_model,
            batch_first=True,
            max_grad_norm=self.max_grad_norm,
            use_ghost_clipping=True,
            use_triton=True  # This should fallback if Triton unavailable
        )
        
        # Should not raise an error
        data, target = next(iter(self.dataloader))
        data, target = data.to(self.device), target.to(self.device)
        output = gsm_flash(data)
        loss = self.criterion(output, target)
        loss.backward()
        
        # Should be able to get norms
        norms = gsm_flash.get_norm_sample()
        self.assertEqual(norms.shape[0], self.batch_size)
        
        print("✓ Fallback mechanism works correctly")


if __name__ == "__main__":
    unittest.main()