#!/usr/bin/env python3
"""
Wrapper for DiT model to make it compatible with Opacus DP-SGD.
The key insight: conditional inputs (timestep, labels) must be handled specially.
"""

import torch
import torch.nn as nn


class DiTDPWrapper(nn.Module):
    """
    Wrapper that makes DiT compatible with Opacus by storing conditional inputs
    as module state rather than forward arguments.
    """
    def __init__(self, dit_model):
        super().__init__()
        self.dit_model = dit_model
        self.timesteps = None
        self.labels = None
    
    def set_conditioning(self, timesteps, labels):
        """Set the conditioning inputs before forward pass"""
        self.timesteps = timesteps
        self.labels = labels
    
    def forward(self, x):
        """
        Forward pass with stored conditioning.
        Args:
            x: images (B, C, H, W)
        Returns:
            predicted_noise (B, C, H, W)
        """
        if self.timesteps is None or self.labels is None:
            raise RuntimeError("Must call set_conditioning() before forward()")
        
        return self.dit_model(x, self.timesteps, self.labels, target_noise=None)


if __name__ == "__main__":
    from diffusion_dit_model import DiTModelWithFlashAttention
    
    # Test the wrapper
    dit = DiTModelWithFlashAttention(
        img_size=256,
        patch_size=8,
        in_channels=3,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        num_classes=1000,
    )
    
    wrapper = DiTDPWrapper(dit)
    
    # Set conditioning
    batch_size = 2
    timesteps = torch.randint(0, 1000, (batch_size,))
    labels = torch.randint(0, 1000, (batch_size,))
    wrapper.set_conditioning(timesteps, labels)
    
    # Forward
    images = torch.randn(batch_size, 3, 256, 256)
    output = wrapper(images)
    
    print(f"✓ Wrapper test successful!")
    print(f"  Input shape: {images.shape}")
    print(f"  Output shape: {output.shape}")

