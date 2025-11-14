#!/usr/bin/env python3
"""
Small transformer model for testing Flash Clipping with FSDP.
Uses DPMultiheadAttentionWithFlashAttention for efficient DP training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from memory_test.test_algo.memory_profile_with_flash_attention import (
    DPMultiheadAttentionWithFlashAttention,
)


class TransformerBlockDP(nn.Module):
    """
    Single transformer block with DP-compatible Flash Attention.
    """
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Self-attention with Flash Attention
        self.attn = DPMultiheadAttentionWithFlashAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=False,  # No bias to avoid FSDP replication issues
            batch_first=True,
        )
        
        # Feed-forward network
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_dim, bias=False),
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
        Returns:
            output: [batch_size, seq_len, hidden_dim]
        """
        # Self-attention with residual
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        
        # FFN with residual
        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)
        
        return x


class SmallTransformerDP(nn.Module):
    """
    Small transformer model for testing (~10M parameters).
    Architecture:
    - Token embedding
    - Positional embedding
    - 3 transformer blocks
    - Classification head
    """
    def __init__(
        self,
        vocab_size=10000,
        hidden_dim=512,
        num_heads=8,
        num_layers=3,
        num_classes=10,
        max_seq_len=128,
        dropout=0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # Embeddings (using nn.Embedding to avoid parameter ownership issues)
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlockDP(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Output head
        self.ln_final = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_ids):
        """
        Args:
            input_ids: [batch_size, seq_len] - token indices
        Returns:
            logits: [batch_size, num_classes]
        """
        batch_size, seq_len = input_ids.shape
        
        # Generate position ids
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embed tokens and positions
        token_embeds = self.token_embedding(input_ids)  # [B, L, D]
        pos_embeds = self.pos_embedding(pos_ids)  # [B, L, D]
        x = self.dropout(token_embeds + pos_embeds)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_final(x)
        
        # Pool by taking mean over sequence dimension
        x = x.mean(dim=1)  # [B, D]
        
        # Classification head
        logits = self.head(x)  # [B, num_classes]
        
        return logits
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_synthetic_dataset(num_samples=1000, vocab_size=10000, seq_len=64, num_classes=10, seed=42):
    """
    Create a synthetic dataset for testing.
    
    Args:
        num_samples: Number of samples
        vocab_size: Vocabulary size
        seq_len: Sequence length
        num_classes: Number of classes
        seed: Random seed
    
    Returns:
        inputs: [num_samples, seq_len] - token indices
        labels: [num_samples] - class labels
    """
    torch.manual_seed(seed)
    inputs = torch.randint(0, vocab_size, (num_samples, seq_len))
    labels = torch.randint(0, num_classes, (num_samples,))
    return inputs, labels


if __name__ == "__main__":
    # Test model creation
    model = SmallTransformerDP(
        vocab_size=10000,
        hidden_dim=512,
        num_heads=8,
        num_layers=3,
        num_classes=10,
        max_seq_len=128,
    )
    
    print(f"Model created with {model.count_parameters():,} parameters")
    
    # Test forward pass
    batch_size = 4
    seq_len = 64
    input_ids = torch.randint(0, 10000, (batch_size, seq_len))
    
    output = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test dataset creation
    inputs, labels = create_synthetic_dataset(num_samples=100, seq_len=64)
    print(f"Dataset created: inputs {inputs.shape}, labels {labels.shape}")

