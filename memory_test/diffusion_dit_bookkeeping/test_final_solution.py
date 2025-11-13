#!/usr/bin/env python3
"""
Final solution test: DiT + DP-SGD with wrapper and manual loss computation
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
from memory_test.diffusion_dit_bookkeeping.dit_dp_wrapper import DiTDPWrapper


def test_final_solution():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    
    print("="*80)
    print("FINAL SOLUTION: DiT + DP-SGD")
    print("="*80)
    
    # Create DiT model
    dit_model = DiTModelWithFlashAttention(
        img_size=256,
        patch_size=8,
        in_channels=3,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        num_classes=1000,
    ).to(device)
    
    # Wrap for DP compatibility
    model = DiTDPWrapper(dit_model).to(device)
    print(f"✓ Model created and wrapped: {dit_model.count_parameters():,} parameters")
    
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
    
    print("\n" + "="*80)
    print("Running training iterations")
    print("="*80)
    
    for iteration in range(3):
        print(f"\nIteration {iteration + 1}:")
        
        images = torch.randn(batch_size, 3, 256, 256, device=device)
        timesteps = torch.randint(0, 1000, (batch_size,), device=device)
        labels = torch.randint(0, 1000, (batch_size,), device=device)
        target_noise = torch.randn_like(images)
        
        # Set conditioning
        model._module.set_conditioning(timesteps, labels)
        
        # Forward
        predicted_noise = model(images)
        if predicted_noise.shape[1] > 3:
            predicted_noise = predicted_noise[:, :3, :, :]
        
        # Compute per-sample MSE loss
        loss_per_sample = F.mse_loss(predicted_noise, target_noise, reduction='none')
        loss_per_sample = loss_per_sample.view(batch_size, -1).mean(dim=1)
        loss = loss_per_sample.mean()
        
        print(f"  Loss: {loss.item():.4f}")
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"  ✓ Iteration {iteration + 1} successful!")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nSolution Summary:")
    print("1. Use DiTDPWrapper to handle conditional inputs (timestep, labels)")
    print("2. Compute loss manually (don't use DPLossFastGradientClipping for images)")
    print("3. Reduce loss to per-sample then to scalar before backward()")


if __name__ == "__main__":
    test_final_solution()

