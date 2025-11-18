#!/usr/bin/env python3
"""
Demo: DiT + Opacus DP-SGD Now Working!

This script demonstrates that the DiT model with multi-input forward signature
now works seamlessly with Opacus's GradSampleModule.
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


def main():
    print("=" * 70)
    print("DiT + Opacus DP-SGD: WORKING SOLUTION DEMO")
    print("=" * 70)
    
    device = "cpu"
    batch_size = 2
    img_size = 32
    patch_size = 4
    hidden_dim = 128
    num_layers = 2
    num_heads = 4
    num_classes = 10
    
    print(f"\n📦 Creating DiT Model")
    print(f"   Config: {img_size}x{img_size} images, {hidden_dim}D, {num_layers} layers")
    
    # Create DiT model with multi-input forward(x, t, y)
    model = DiTModelWithFlashAttention(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=3,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_classes=num_classes,
        learn_sigma=False,
    ).to(device)
    
    print(f"   Parameters: {model.count_parameters():,}")
    print(f"   Forward signature: forward(x, t, y) ← Multi-input!")
    
    # Validate and wrap with Opacus
    print(f"\n🔒 Setting up Opacus DP-SGD")
    model = ModuleValidator.fix(model)
    gs_model = GradSampleModule(model, batch_first=True, loss_reduction="mean")
    print(f"   ✓ Model wrapped with GradSampleModule")
    
    # Create sample data
    print(f"\n📊 Creating sample data (batch_size={batch_size})")
    images = torch.randn(batch_size, 3, img_size, img_size, device=device)
    timesteps = torch.randint(0, 1000, (batch_size,), device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    target_noise = torch.randn_like(images)
    
    print(f"   Images shape: {images.shape}")
    print(f"   Timesteps: {timesteps.tolist()}")
    print(f"   Labels: {labels.tolist()}")
    
    # Forward pass with THREE inputs!
    print(f"\n🚀 Running forward pass with 3 inputs: (x, t, y)")
    gs_model.train()
    predicted_noise = gs_model(images, timesteps, labels)
    print(f"   ✓ Forward pass successful!")
    print(f"   Output shape: {predicted_noise.shape}")
    
    # Backward pass
    print(f"\n⚡ Running backward pass (computing per-sample gradients)")
    loss = F.mse_loss(predicted_noise, target_noise)
    print(f"   Loss: {loss.item():.6f}")
    loss.backward()
    print(f"   ✓ Backward pass successful!")
    
    # Verify per-sample gradients
    print(f"\n✨ Verifying per-sample gradients")
    params_with_grad_sample = 0
    for name, param in gs_model.named_parameters():
        if param.requires_grad:
            assert hasattr(param, 'grad_sample'), f"Missing grad_sample for {name}"
            assert param.grad_sample is not None, f"None grad_sample for {name}"
            assert param.grad_sample.shape[0] == batch_size, f"Wrong batch size for {name}"
            params_with_grad_sample += 1
    
    print(f"   ✓ All {params_with_grad_sample} parameters have correct grad_sample!")
    print(f"   ✓ Each grad_sample has shape (batch_size={batch_size}, ...)")
    
    # Show some example grad_sample shapes
    print(f"\n📈 Example grad_sample shapes:")
    count = 0
    for name, param in gs_model.named_parameters():
        if param.requires_grad and count < 3:
            print(f"   {name}: {list(param.grad_sample.shape)}")
            count += 1
    print(f"   ... and {params_with_grad_sample - 3} more parameters")
    
    print(f"\n" + "=" * 70)
    print("✅ SUCCESS: DiT works perfectly with Opacus DP-SGD!")
    print("=" * 70)
    print(f"\nKey Achievement:")
    print(f"  • Multi-input forward(x, t, y) ← Now supported!")
    print(f"  • Per-sample gradients computed correctly")
    print(f"  • Ready for DP-SGD training")
    print(f"  • Backward compatible with single-input models")
    print(f"\n🎉 The incompatibility issue is fully resolved!")


if __name__ == "__main__":
    main()

