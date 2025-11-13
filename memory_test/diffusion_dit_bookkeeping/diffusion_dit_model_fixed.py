#!/usr/bin/env python3
"""
FIXED DiT Model that works with Opacus DP-SGD.

Key changes:
1. Conditional inputs (timestep, labels) are passed as part of the input tensor
2. Forward signature is simplified to forward(x) where x contains all necessary information
3. Compatible with Opacus's functorch-based per-sample gradient computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))

from memory_test.diffusion_dit_bookkeeping.diffusion_dit_model import (
    TimestepEmbedder,
    LabelEmbedder,
    PatchEmbed,
    DiTBlockWithFlashAttention,
    FinalLayer,
)


class DiTModelFixedForDP(nn.Module):
    """
    DiT model compatible with Opacus DP-SGD.
    
    Key difference: forward() takes a single concatenated input instead of separate arguments.
    Input format: torch.cat([images, timesteps_expanded, labels_expanded], dim=1)
    - images: (B, 3, H, W)
    - timesteps_expanded: (B, 1, H, W) - timestep ID repeated spatially
    - labels_expanded: (B, 1, H, W) - label ID repeated spatially
    Total input channels: 3 + 1 + 1 = 5
    """
    
    def __init__(
        self,
        img_size=256,
        patch_size=8,
        in_channels=3,  # Still 3 for RGB, we handle conditionals separately
        hidden_dim=1024,
        num_layers=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Patch embedding for COMBINED input (images + timestep_map + label_map)
        # Input will be (B, 5, H, W) where channels are [R, G, B, timestep_id, label_id]
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels=5, embed_dim=hidden_dim)
        num_patches = self.patch_embed.num_patches
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        
        # Timestep and label embedders
        # These process the scalar IDs extracted from the input
        self.timestep_embedder = TimestepEmbedder(hidden_dim)
        self.label_embedder = LabelEmbedder(num_classes, hidden_dim, dropout_prob=0.1)
        
        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlockWithFlashAttention(hidden_dim, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(num_layers)
        ])
        
        # Final layer
        self.final_layer = FinalLayer(hidden_dim, patch_size, self.out_channels)
        
        # Initialize weights
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize patch_embed
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embed.proj.bias, 0)
        
        # Initialize positional embedding
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Initialize timestep embedding MLP
        nn.init.normal_(self.timestep_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp[2].weight, std=0.02)
        
        # Initialize label embedding
        nn.init.normal_(self.label_embedder.embedding_table.weight, std=0.02)
        
        # Initialize transformer blocks
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Zero-initialize adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation.linear.weight, 0)
            nn.init.constant_(block.adaLN_modulation.linear.bias, 0)
        
        # Zero-initialize final layer
        nn.init.constant_(self.final_layer.adaLN_modulation.linear.weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation.linear.bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def unpatchify(self, x):
        """
        x: (B, N, patch_size^2 * C)
        imgs: (B, C, H, W)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def forward(self, x_combined):
        """
        Forward pass compatible with Opacus.
        
        Args:
            x_combined: concatenated input (B, 5, H, W)
                - channels 0-2: RGB image
                - channel 3: timestep ID (repeated spatially)
                - channel 4: class label ID (repeated spatially)
        
        Returns:
            predicted_noise (B, out_channels, H, W)
        """
        # Split combined input
        images = x_combined[:, :3, :, :]  # (B, 3, H, W)
        timestep_map = x_combined[:, 3:4, :, :]  # (B, 1, H, W)
        label_map = x_combined[:, 4:5, :, :]  # (B, 1, H, W)
        
        # Extract scalar timestep and label from the spatial maps
        # (they're constant across spatial dimensions)
        t = timestep_map[:, 0, 0, 0].long()  # (B,)
        y = label_map[:, 0, 0, 0].long()  # (B,)
        
        # Patch embedding (processes all 5 channels)
        x = self.patch_embed(x_combined)  # (B, N, hidden_dim)
        x = x + self.pos_embed
        
        # Timestep and label conditioning
        t_emb = self.timestep_embedder(t)  # (B, hidden_dim)
        y_emb = self.label_embedder(y, self.training)  # (B, hidden_dim)
        c = t_emb + y_emb  # (B, hidden_dim)
        
        # Apply DiT blocks
        for block in self.blocks:
            x = block(x, c)
        
        # Final layer
        x = self.final_layer(x, c)  # (B, N, patch_size^2 * out_channels)
        
        # Unpatchify
        predicted_noise = self.unpatchify(x)  # (B, out_channels, H, W)
        
        return predicted_noise
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def prepare_dit_input(images, timesteps, labels):
    """
    Prepare input tensor for DiTModelFixedForDP.
    
    Args:
        images: (B, 3, H, W)
        timesteps: (B,) - scalar timestep IDs
        labels: (B,) - scalar class labels
    
    Returns:
        x_combined: (B, 5, H, W) - concatenated input
    """
    B, C, H, W = images.shape
    
    # Expand timesteps and labels to spatial maps
    timestep_map = timesteps.view(B, 1, 1, 1).expand(B, 1, H, W).float()
    label_map = labels.view(B, 1, 1, 1).expand(B, 1, H, W).float()
    
    # Concatenate
    x_combined = torch.cat([images, timestep_map, label_map], dim=1)  # (B, 5, H, W)
    
    return x_combined


if __name__ == "__main__":
    # Test the fixed model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = DiTModelFixedForDP(
        img_size=256,
        patch_size=8,
        in_channels=3,
        hidden_dim=1024,
        num_layers=24,
        num_heads=16,
        num_classes=1000,
    ).to(device)
    
    print(f"✓ DiT Fixed Model created")
    print(f"  Parameters: {model.count_parameters():,}")
    
    # Test with combined input
    batch_size = 2
    images = torch.randn(batch_size, 3, 256, 256, device=device)
    timesteps = torch.randint(0, 1000, (batch_size,), device=device)
    labels = torch.randint(0, 1000, (batch_size,), device=device)
    
    # Prepare input
    x_combined = prepare_dit_input(images, timesteps, labels)
    print(f"✓ Combined input shape: {x_combined.shape}")
    
    # Forward pass
    output = model(x_combined)
    print(f"✓ Forward pass successful")
    print(f"  Output shape: {output.shape}")
    
    # Test backward
    target = torch.randn_like(output)
    loss = F.mse_loss(output, target)
    loss.backward()
    print(f"✓ Backward pass successful")
    print(f"  Loss: {loss.item():.4f}")

