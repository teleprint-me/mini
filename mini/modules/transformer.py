"""
Copyright © 2023 Austin Berrio
Module: mini.modules.transformer
Description: Contains the hidden transformer model components.
"""

import torch.nn as nn
from torch.nn import LayerNorm

from mini.config import ConfigTransformer
from mini.modules.attention import RotaryAttention, SelfAttention
from mini.modules.feed_forward import GatedFeedForward, PositionWiseFeedForward
from mini.modules.normalization import RMSNorm


class PositionWiseBlock(nn.Module):
    """Standard Transformer Block using Position-Wise FFN and Self-Attention."""

    def __init__(self, config: ConfigTransformer):
        super().__init__()
        self.attn = SelfAttention(config)
        self.norm1 = LayerNorm(normalized_shape=config.embed_dim)
        self.norm2 = LayerNorm(normalized_shape=config.embed_dim)
        self.ff = PositionWiseFeedForward(config)
        self.dropout = nn.Dropout(config.dropout)  # Dropout before residuals

    def forward(self, x, mask=None):
        """Forward pass through the transformer block."""
        x = self.norm1(x + self.dropout(self.attn(x, mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class GatedBlock(nn.Module):
    """Transformer Block using Gated Feed-Forward Network and Rotary Attention."""

    def __init__(self, config: ConfigTransformer):
        super().__init__()
        self.attn = RotaryAttention(config)
        self.norm1 = RMSNorm(config=config)
        self.norm2 = RMSNorm(config=config)
        self.ff = GatedFeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        """Forward pass through the transformer block."""
        x = self.norm1(x + self.dropout(self.attn(x, mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x
