#!/usr/bin/env python3
"""
Diffusion Transformer (DiT) Model with Flash Attention support.
Implements DiT-L architecture for memory profiling experiments.
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

from memory_test.test_algo.memory_profile_with_flash_attention import DPMultiheadAttentionWithFlashAttention


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Uses sinusoidal position embedding followed by MLP.
    """
    def __init__(self, hidden_dim, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_dim, dropout_prob=0.1):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_dim)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train=True, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embedding stored in a module to avoid blocking Opacus recursion.
    Simple container for a parameter that doesn't have trainable operations.
    """
    def __init__(self, num_positions, embed_dim):
        super().__init__()
        self.num_positions = num_positions
        self.embed_dim = embed_dim
        # Store as buffer, not parameter, so grad sampler doesn't try to compute per-sample gradients
        # Positional embeddings are shared across all samples, so per-sample gradients don't make sense
        self.register_buffer('pos_embed', torch.zeros(1, num_positions, embed_dim))
    
    def forward(self, device=None):
        """
        Returns positional embeddings for all positions.
        Returns:
            pos_embed: (1, num_positions, embed_dim)
        """
        return self.pos_embed


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding using Conv2d.
    """
    def __init__(self, img_size=256, patch_size=8, in_channels=3, embed_dim=1024):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        x = self.proj(x)  # (B, embed_dim, grid_size, grid_size)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class AdaLNModulation(nn.Module):
    """
    Adaptive Layer Normalization (adaLN) modulation.
    Outputs scale and shift parameters for layer normalization based on conditioning.
    """
    def __init__(self, hidden_dim, num_outputs=6):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_outputs * hidden_dim, bias=True)
        self.num_outputs = num_outputs

    def forward(self, c):
        """
        Args:
            c: conditioning vector (B, hidden_dim)
        Returns:
            tuple of (num_outputs) tensors, each (B, hidden_dim)
        """
        out = self.linear(F.silu(c))
        return out.chunk(self.num_outputs, dim=-1)


class DiTBlockWithFlashAttention(nn.Module):
    """
    DiT block with Flash Attention and adaptive layer normalization (adaLN-Zero).
    """
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = DPMultiheadAttentionWithFlashAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.0,
            bias=True,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_dim, bias=True),
        )
        # adaLN modulation: outputs 6 parameters (scale/shift for norm1, norm2, and gate for attn/mlp)
        self.adaLN_modulation = AdaLNModulation(hidden_dim, num_outputs=6)

    def forward(self, x, c):
        """
        Args:
            x: input tensor (B, N, hidden_dim)
            c: conditioning vector (B, hidden_dim)
        Returns:
            output tensor (B, N, hidden_dim)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)
        
        # Self-attention with adaLN
        x_norm = self.norm1(x)
        x_modulated = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_output, _ = self.attn(x_modulated, x_modulated, x_modulated)
        x = x + gate_msa.unsqueeze(1) * attn_output
        
        # MLP with adaLN
        x_norm = self.norm2(x)
        x_modulated = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_output = self.mlp(x_modulated)
        x = x + gate_mlp.unsqueeze(1) * mlp_output
        
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT that outputs the predicted noise.
    """
    def __init__(self, hidden_dim, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = AdaLNModulation(hidden_dim, num_outputs=2)

    def forward(self, x, c):
        """
        Args:
            x: input tensor (B, N, hidden_dim)
            c: conditioning vector (B, hidden_dim)
        Returns:
            output tensor (B, N, patch_size^2 * out_channels)
        """
        shift, scale = self.adaLN_modulation(c)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x


class DiTModelWithFlashAttention(nn.Module):
    """
    Diffusion Transformer (DiT) model with Flash Attention.
    DiT-L configuration: hidden_dim=1024, num_layers=24, num_heads=16
    """
    def __init__(
        self,
        img_size=256,
        patch_size=8,
        in_channels=3,
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

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, hidden_dim)
        num_patches = self.patch_embed.num_patches

        # Positional embedding (learnable)
        # Use custom PositionalEmbedding module instead of nn.Parameter to avoid blocking iterate_submodules in Opacus
        # When the top-level module has direct parameters, Opacus won't recurse into submodules
        self.pos_embed_module = PositionalEmbedding(num_patches, hidden_dim)

        # Timestep embedding
        self.timestep_embedder = TimestepEmbedder(hidden_dim)

        # Label embedding (for class-conditional generation)
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
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embed.proj.bias, 0)

        # Initialize positional embedding (stored as buffer)
        # Positional embeddings are NOT trained with DP-SGD (shared across all samples)
        nn.init.normal_(self.pos_embed_module.pos_embed, std=0.02)

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

        # Zero-initialize adaLN modulation layers (adaLN-Zero)
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

    def forward(self, x, t, y, target_noise=None):
        """
        Forward pass of DiT.
        Args:
            x: input images (B, C, H, W)
            t: diffusion timesteps (B,)
            y: class labels (B,)
            target_noise: optional target noise for computing loss (B, C, H, W)
        Returns:
            dict with 'logits' (predicted noise) and optionally 'loss'
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, hidden_dim)
        # Add positional embedding
        pos_embed = self.pos_embed_module(x.device)  # (1, N, hidden_dim)
        x = x + pos_embed

        # Timestep and label conditioning
        t_emb = self.timestep_embedder(t)  # (B, hidden_dim)
        y_emb = self.label_embedder(y, self.training)  # (B, hidden_dim)
        c = t_emb + y_emb  # Conditioning vector (B, hidden_dim)

        # Apply DiT blocks
        for block in self.blocks:
            x = block(x, c)

        # Final layer
        x = self.final_layer(x, c)  # (B, N, patch_size^2 * out_channels)

        # Unpatchify to get predicted noise
        predicted_noise = self.unpatchify(x)  # (B, out_channels, H, W)

        # Compute loss if target noise is provided
        if target_noise is not None:
            # For learn_sigma=True, split prediction into noise and variance
            if self.out_channels > self.in_channels:
                predicted_noise_only = predicted_noise[:, :self.in_channels, :, :]
            else:
                predicted_noise_only = predicted_noise
            loss = F.mse_loss(predicted_noise_only, target_noise)
            
            return {
                "logits": predicted_noise,
                "loss": loss
            }
        else:
            # When target_noise is None (DP-SGD mode), return just the tensor
            # to avoid issues with backward hooks expecting Tensor, not dict
            return predicted_noise

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # DiT-L configuration
    model = DiTModelWithFlashAttention(
        img_size=256,
        patch_size=8,
        in_channels=3,
        hidden_dim=1024,
        num_layers=24,
        num_heads=16,
        num_classes=1000,
    ).to(device)
    
    print(f"DiT-L Model created")
    print(f"Number of parameters: {model.count_parameters():,}")
    print(f"Number of tokens: {model.patch_embed.num_patches}")
    
    # Test forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 256, 256, device=device)
    timesteps = torch.randint(0, 1000, (batch_size,), device=device)
    labels = torch.randint(0, 1000, (batch_size,), device=device)
    target_noise = torch.randn_like(images)
    
    output = model(images, timesteps, labels, target_noise)
    print(f"Output shape: {output['logits'].shape}")
    print(f"Loss: {output['loss'].item():.4f}")

