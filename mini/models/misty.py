"""
Copyright © 2023 Austin Berrio
Module: mini.models.misty
Author: Austin Berrio
Date: 2025-02-07
Version: 0.1.0
License: AGPL License
URL: https://github.com/teleprint-me/mini
Description:
Misty is a minimal transformer model designed for educational purposes.
It is a simplified version of the transformer architecture, suitable for beginners
to understand the basics of neural network design and training.
"""

import torch
import torch.nn as nn

from mini.config import ConfigTransformer
from mini.modules.attention import AttentionMask
from mini.modules.embedding import PositionalEmbedding
from mini.modules.transformer import PositionWiseBlock


# Attention is All You Need: https://arxiv.org/abs/1706.03762
class MistyModel(nn.Module):
    """Minimal positional transformer model with self-attention and FFN layers."""

    def __init__(self, config: ConfigTransformer):
        super().__init__()
        self.bias = config.bias

        # Mask is programmatically moved to the correct device and dtype defined by the config
        self.mask = AttentionMask(config)
        self.embedding = PositionalEmbedding(config)
        self.blocks = nn.ModuleList(
            [PositionWiseBlock(config) for _ in range(config.num_layers)]
        )
        self.norm = nn.LayerNorm(normalized_shape=config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size)

        self.max_seq_len = config.max_seq_len
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        nn.init.xavier_uniform_(self.head.weight)
        if self.bias:
            nn.init.uniform_(self.head.bias, a=-0.1, b=0.1)

    def forward(self, x: torch.Tensor, mask_type: str = "causal") -> torch.Tensor:
        """Forward pass through Misty transformer."""
        # Combine padding and attention masks
        # NOTE: # This must happen before the embeddings layer
        mask = self.mask(x, mask_type=mask_type)
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        return self.head(x)
