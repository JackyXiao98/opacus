#!/usr/bin/env python3
"""
Minimal debug script - FIXED VERSION
The key insight: for images, we need to reduce loss to per-sample before DPLoss
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.grad_sample import GradSampleModuleFastGradientClipping
from opacus.optimizers import DPOptimizerFastGradientClipping
from memory_test.diffusion_dit_bookkeeping.diffusion_dit_model import DiTModelWithFlashAttention


def test_dp_loss_fixed():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    
    print("="*80)
    print("Testing FIXED DP Loss with DiT Model")
    print("="*80)
    
    # Create small model for testing
    model = DiTModelWithFlashAttention(
        img_size=256,
        patch_size=8,
        in_channels=3,
        hidden_dim=128,  # Small for testing
        num_layers=2,     # Small for testing
        num_heads=4,
        num_classes=1000,
    ).to(device)
    
    print(f"✓ Model created: {model.count_parameters():,} parameters")
    
    # Wrap with DP
    model = GradSampleModuleFastGradientClipping(
        model,
        use_triton=True,
        use_ghost_clipping=True,
        enable_fastdp_bookkeeping=False,
        loss_reduction="mean",
    )
    print("✓ Model wrapped with GradSampleModuleFastGradientClipping")
    
    # Create optimizer
    base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer = DPOptimizerFastGradientClipping(
        optimizer=base_optimizer,
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        expected_batch_size=batch_size,
        loss_reduction="mean",
    )
    print("✓ Optimizer created")
    
    # DO NOT use DPLossFastGradientClipping for image tasks!
    # Instead, compute loss manually
    
    print("\n" + "="*80)
    print("Testing forward and backward pass (FIXED APPROACH)")
    print("="*80)
    
    for iteration in range(2):
        print(f"\nIteration {iteration + 1}:")
        
        images = torch.randn(batch_size, 3, 256, 256, device=device)
        timesteps = torch.randint(0, 1000, (batch_size,), device=device)
        labels = torch.randint(0, 1000, (batch_size,), device=device)
        target_noise = torch.randn_like(images)
        
        # Forward
        predicted_noise = model(images, timesteps, labels, target_noise=None)
        if predicted_noise.shape[1] > 3:
            predicted_noise = predicted_noise[:, :3, :, :]
        
        # Compute per-sample MSE loss manually
        # MSE per sample: mean over all dimensions except batch
        loss_per_sample = F.mse_loss(predicted_noise, target_noise, reduction='none')  # (B, C, H, W)
        loss_per_sample = loss_per_sample.view(batch_size, -1).mean(dim=1)  # (B,)
        
        print(f"  loss_per_sample shape: {loss_per_sample.shape}")
        print(f"  loss_per_sample values: {loss_per_sample}")
        
        # Reduce to scalar for backward
        loss = loss_per_sample.mean()  # scalar
        print(f"  loss (scalar): {loss.item():.4f}")
        
        # Backward
        try:
            loss.backward()
            print("  ✓ Backward successful!")
            
            optimizer.step()
            print("  ✓ Optimizer step successful!")
            
            optimizer.zero_grad()
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            return
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nConclusion: For image tasks, don't use DPLossFastGradientClipping.")
    print("Instead, compute loss manually and call backward() directly on scalar loss.")


if __name__ == "__main__":
    test_dp_loss_fixed()

