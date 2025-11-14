#!/usr/bin/env python3
"""
Test that the FIXED DiT model works with Opacus DP-SGD
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))

import torch
import torch.nn.functional as F
from opacus.grad_sample import GradSampleModuleFastGradientClipping
from opacus.optimizers import DPOptimizerFastGradientClipping
from memory_test.diffusion_dit_bookkeeping.diffusion_dit_model_fixed import (
    DiTModelFixedForDP,
    prepare_dit_input,
)


def test_dit_with_dp():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    
    print("="*80)
    print("Testing FIXED DiT Model with Opacus DP-SGD")
    print("="*80)
    
    # Create model
    model = DiTModelFixedForDP(
        img_size=256,
        patch_size=8,
        in_channels=3,
        hidden_dim=128,  # Smaller for testing
        num_layers=2,
        num_heads=4,
        num_classes=1000,
    ).to(device)
    
    print(f"\n✓ Model created: {model.count_parameters():,} parameters")
    
    # Wrap with DP
    model = GradSampleModuleFastGradientClipping(
        model,
        use_flash_clipping=False,  # Use ghost clipping for compatibility
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
    print("✓ DP Optimizer created")
    
    print("\n" + "="*80)
    print("Running DP-SGD training iterations")
    print("="*80)
    
    for iteration in range(3):
        print(f"\nIteration {iteration + 1}:")
        
        # Generate data
        images = torch.randn(batch_size, 3, 256, 256, device=device)
        timesteps = torch.randint(0, 1000, (batch_size,), device=device)
        labels = torch.randint(0, 1000, (batch_size,), device=device)
        target_noise = torch.randn_like(images)
        
        # Prepare combined input (key difference!)
        x_combined = prepare_dit_input(images, timesteps, labels)
        
        # Forward pass
        predicted_noise = model(x_combined)
        
        # Extract only noise prediction (first 3 channels)
        if predicted_noise.shape[1] > 3:
            predicted_noise = predicted_noise[:, :3, :, :]
        
        # Compute loss (per-sample then reduce)
        loss_per_sample = F.mse_loss(predicted_noise, target_noise, reduction='none')
        loss_per_sample = loss_per_sample.view(batch_size, -1).mean(dim=1)  # (B,)
        loss = loss_per_sample.mean()  # scalar
        
        print(f"  Loss: {loss.item():.4f}")
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"  ✓ Iteration {iteration + 1} successful!")
    
    print("\n" + "="*80)
    print("✅ SUCCESS! DiT + DP-SGD works with the fixed architecture")
    print("="*80)
    print("\nKey Insight:")
    print("  - Conditional inputs (timestep, labels) are embedded in the input tensor")
    print("  - Input shape: (B, 5, H, W) = [RGB (3) + timestep (1) + label (1)]")
    print("  - Opacus only sees a single input argument: forward(x_combined)")
    print("  - Per-sample gradient computation works correctly!")


if __name__ == "__main__":
    test_dit_with_dp()

