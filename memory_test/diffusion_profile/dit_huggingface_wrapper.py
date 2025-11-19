#!/usr/bin/env python3
"""
DiT Model Wrapper for DP-SGD Experiments

This module provides a DiT model wrapper that uses DP-compatible attention layers
for Opacus DP-SGD framework. While it references HuggingFace model configs for
architecture specifications, it uses a custom implementation with per-sample
gradient computation support.
"""

import torch
import torch.nn as nn
from typing import Optional
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))


class DiTHuggingFaceWrapper(nn.Module):
    """
    DP-compatible DiT model wrapper.
    
    This wrapper uses a custom DiT implementation with DP-compatible attention layers
    instead of HuggingFace models (which use standard attention layers that don't
    support per-sample gradient computation).
    
    Features:
    1. Uses DPMultiheadAttentionWithFlashAttention for efficient per-sample gradients
    2. Handles forward pass with diffusion inputs (images, timesteps, labels)
    3. Returns dict with loss when target_noise is provided (vanilla training)
    4. Returns tensor when target_noise is None (DP-SGD mode)
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/dit-large",  # Using a valid DiT model
        img_size: int = 1024,
        patch_size: int = 8,
        in_channels: int = 3,
        num_classes: int = 1000,
        pretrained: bool = False,
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            img_size: Input image size (height and width)
            patch_size: Patch size for tokenization
            in_channels: Number of input channels
            num_classes: Number of label classes
            pretrained: Whether to load pretrained weights
        """
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Use DP-compatible fallback model instead of HuggingFace
        # HuggingFace models use standard attention layers that don't support
        # per-sample gradient computation needed for DP-SGD
        print(f"Creating DP-compatible DiT model (config reference: {model_name})")
        self.dit_model = self._create_fallback_model()
        
        # No need for output projection - fallback model handles everything
    
    def _create_fallback_model(self):
        """Create a fallback transformer-based model"""
        from memory_test.diffusion_dit_bookkeeping.diffusion_dit_model import DiTModelWithFlashAttention
        
        return DiTModelWithFlashAttention(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            hidden_dim=1024,  # DiT-Large default
            num_layers=24,
            num_heads=16,
            num_classes=self.num_classes,
        )
    
    def forward(
        self,
        images: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor,
        target_noise: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for diffusion training.
        
        Args:
            images: Input images [B, C, H, W]
            timesteps: Diffusion timesteps [B]
            labels: Class labels [B]
            target_noise: Target noise for loss computation [B, C, H, W] (optional)
        
        Returns:
            When target_noise is provided: dict with 'loss' and 'logits'
            When target_noise is None: tensor [B, C, H, W] (DP-SGD mode)
        """
        # Always use the DP-compatible fallback model
        return self.dit_model(images, timesteps, labels, target_noise)


def create_dit_huggingface_model(
    img_size: int = 1024,
    patch_size: int = 8,
    in_channels: int = 3,
    num_classes: int = 1000,
    pretrained: bool = False,
    device: str = "cuda",
) -> DiTHuggingFaceWrapper:
    """
    Factory function to create a DiT HuggingFace model.
    
    Args:
        img_size: Input image size
        patch_size: Patch size
        in_channels: Number of input channels
        num_classes: Number of classes
        pretrained: Whether to load pretrained weights
        device: Device to move model to
    
    Returns:
        DiTHuggingFaceWrapper instance
    """
    model = DiTHuggingFaceWrapper(
        model_name="microsoft/dit-large",
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=pretrained,
    )
    
    return model.to(device)

