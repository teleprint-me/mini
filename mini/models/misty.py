"""
Copyright © 2023 Austin Berrio

Module: mini.models.misty
Author: Austin Berrio
Date: 2025-02-07
Version: 0.1.0
License: AGPL License
URL: https://github.com/teleprint-me/mini

Description:
Misty is a minimal positional transformer model with positional embeddings and
position-wise feed-forward networks (FFN) designed for educational purposes.
It is a simplified version of the original transformer architecture using basic
self-attention, suitable for beginners to understand the basics of neural network
design and training.
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

        # Programmatic mask type selection, e.g. causal or bidirectional
        self.mask = AttentionMask(config)  # Config-driven mask type
        self.embedding = PositionalEmbedding(config)
        self.transformer = nn.ModuleList(
            [PositionWiseBlock(config) for _ in range(config.num_layers)]
        )
        self.norm = nn.LayerNorm(normalized_shape=config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size)

        self.max_seq_len = config.max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Misty transformer."""
        # NOTE: Apply padding and attention masks before the embeddings layer
        mask = self.mask(x)  # Config-driven device and data type
        x = self.embedding(x)
        for block in self.transformer:
            x = block(x, mask)
        x = self.norm(x)
        return self.head(x)
