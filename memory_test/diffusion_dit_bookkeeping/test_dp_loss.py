#!/usr/bin/env python3
"""
Minimal debug script to test DP loss with DiT model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))

import torch
import torch.nn as nn
from opacus.grad_sample import GradSampleModuleFastGradientClipping
from opacus.optimizers import DPOptimizerFastGradientClipping
from opacus.utils.fast_gradient_clipping_utils import DPLossFastGradientClipping
from memory_test.diffusion_dit_bookkeeping.diffusion_dit_model import DiTModelWithFlashAttention


def test_dp_loss():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    
    print("="*80)
    print("Testing DP Loss with DiT Model")
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
    
    # Create loss
    criterion = nn.MSELoss(reduction="mean")
    dp_loss = DPLossFastGradientClipping(
        model,
        optimizer,
        criterion,
        loss_reduction="mean",
    )
    print("✓ DPLoss created")
    
    # Test forward pass
    print("\n" + "="*80)
    print("Testing forward and backward pass")
    print("="*80)
    
    images = torch.randn(batch_size, 3, 256, 256, device=device)
    timesteps = torch.randint(0, 1000, (batch_size,), device=device)
    labels = torch.randint(0, 1000, (batch_size,), device=device)
    target_noise = torch.randn_like(images)
    
    # Forward
    predicted_noise = model(images, timesteps, labels, target_noise=None)
    print(f"✓ Forward pass successful")
    print(f"  Output shape: {predicted_noise.shape}")
    print(f"  Output type: {type(predicted_noise)}")
    
    # Extract noise prediction
    if predicted_noise.shape[1] > 3:
        predicted_noise = predicted_noise[:, :3, :, :]
        print(f"  Extracted shape: {predicted_noise.shape}")
    
    # Test different loss computation approaches
    print("\n" + "-"*80)
    print("Approach 1: Direct MSE (should work)")
    print("-"*80)
    try:
        # Direct criterion without DPLoss
        loss_direct = criterion(predicted_noise, target_noise)
        print(f"✓ Direct loss computed: {loss_direct.item():.4f}")
        print(f"  Loss shape: {loss_direct.shape}")
        print(f"  Loss is scalar: {loss_direct.dim() == 0}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    print("\n" + "-"*80)
    print("Approach 2: Flatten then DPLoss (current approach)")
    print("-"*80)
    try:
        pred_flat = predicted_noise.view(batch_size, -1)
        target_flat = target_noise.view(batch_size, -1)
        print(f"  Flattened pred shape: {pred_flat.shape}")
        print(f"  Flattened target shape: {target_flat.shape}")
        
        loss = dp_loss(pred_flat, target_flat, shape=(batch_size, pred_flat.shape[1]))
        print(f"✓ DPLoss computed: {loss}")
        print(f"  Loss type: {type(loss)}")
        print(f"  Loss shape: {loss.shape if hasattr(loss, 'shape') else 'N/A'}")
        print(f"  Loss is scalar: {loss.dim() == 0 if hasattr(loss, 'dim') else 'N/A'}")
        
        # Try backward
        print("\n  Attempting backward...")
        loss.backward()
        print("✓ Backward successful!")
        optimizer.step()
        optimizer.zero_grad()
        print("✓ Optimizer step successful!")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "-"*80)
    print("Approach 3: Direct DPLoss without flatten")
    print("-"*80)
    try:
        # Reset model
        optimizer.zero_grad()
        
        # Forward again
        predicted_noise = model(images, timesteps, labels, target_noise=None)
        if predicted_noise.shape[1] > 3:
            predicted_noise = predicted_noise[:, :3, :, :]
        
        loss = dp_loss(predicted_noise, target_noise)
        print(f"✓ DPLoss computed: {loss}")
        print(f"  Loss type: {type(loss)}")
        print(f"  Loss shape: {loss.shape if hasattr(loss, 'shape') else 'N/A'}")
        print(f"  Loss is scalar: {loss.dim() == 0 if hasattr(loss, 'dim') else 'N/A'}")
        
        # Try backward
        print("\n  Attempting backward...")
        loss.backward()
        print("✓ Backward successful!")
        optimizer.step()
        optimizer.zero_grad()
        print("✓ Optimizer step successful!")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_dp_loss()

