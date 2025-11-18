#!/usr/bin/env python3
"""
Quick test to verify the actual DiT model from diffusion_dit_model.py works with Opacus.
"""

import torch
import torch.nn.functional as F
import sys
import os

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))

from opacus import GradSampleModule
from opacus.validators import ModuleValidator
from diffusion_dit_model import DiTModelWithFlashAttention


def test_real_dit_with_opacus():
    """Test that the actual DiT model works with Opacus"""
    print("\n" + "="*70)
    print("Testing Real DiT Model with Opacus DP-SGD")
    print("="*70)
    
    device = "cpu"  # Use CPU for testing (no CUDA required)
    batch_size = 2
    img_size = 32  # Small for fast testing
    patch_size = 4
    hidden_dim = 128  # Small for fast testing
    num_layers = 2
    num_heads = 4
    num_classes = 10
    
    print(f"\nCreating DiT model:")
    print(f"  - Image size: {img_size}x{img_size}")
    print(f"  - Hidden dim: {hidden_dim}")
    print(f"  - Num layers: {num_layers}")
    print(f"  - Batch size: {batch_size}")
    
    # Create model
    model = DiTModelWithFlashAttention(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=3,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_classes=num_classes,
        learn_sigma=False,  # Simplify output
    ).to(device)
    
    print(f"\nModel parameters: {model.count_parameters():,}")
    
    # Validate model for DP
    model = ModuleValidator.fix(model)
    print("✓ Model validated for DP-SGD")
    
    # Wrap with GradSampleModule
    gs_model = GradSampleModule(model, batch_first=True, loss_reduction="mean")
    print("✓ Model wrapped with GradSampleModule")
    
    # Create sample data
    images = torch.randn(batch_size, 3, img_size, img_size, device=device)
    timesteps = torch.randint(0, 1000, (batch_size,), device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    target_noise = torch.randn_like(images)
    
    print(f"\nInput shapes:")
    print(f"  - images: {images.shape}")
    print(f"  - timesteps: {timesteps.shape}")
    print(f"  - labels: {labels.shape}")
    
    # Forward pass
    gs_model.train()
    predicted_noise = gs_model(images, timesteps, labels)
    
    print(f"\nOutput shape: {predicted_noise.shape}")
    assert predicted_noise.shape == images.shape, \
        f"Expected shape {images.shape}, got {predicted_noise.shape}"
    print("✓ Forward pass successful")
    
    # Backward pass
    loss = F.mse_loss(predicted_noise, target_noise)
    print(f"\nLoss: {loss.item():.6f}")
    loss.backward()
    print("✓ Backward pass successful")
    
    # Check grad_sample exists
    grad_sample_count = 0
    for name, param in gs_model.named_parameters():
        if param.requires_grad:
            assert hasattr(param, 'grad_sample'), f"Parameter {name} missing grad_sample"
            assert param.grad_sample is not None, f"Parameter {name} has None grad_sample"
            assert param.grad_sample.shape[0] == batch_size, \
                f"Parameter {name} grad_sample batch dimension is {param.grad_sample.shape[0]}, expected {batch_size}"
            grad_sample_count += 1
    
    print(f"\n✓ All {grad_sample_count} parameters have correct grad_sample")
    
    print("\n" + "="*70)
    print("🎉 SUCCESS: Real DiT model works with Opacus DP-SGD!")
    print("="*70)
    
    return True


if __name__ == "__main__":
    try:
        test_real_dit_with_opacus()
        print("\n✅ Test PASSED")
        exit(0)
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

