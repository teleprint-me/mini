"""
Copyright © 2023 Austin Berrio
Module: mini.transformer.embedding
Description: Embedding blocks for transformer models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini.transformer.config import TransformerConfig
from mini.transformer.encoding import PositionalEncoding


class Embedding(nn.Module):
    """Handles token and positional embeddings for the transformer model."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.pad_id = max(config.pad_id, 0)
        self.embed_dim = config.embed_dim

        # Token Embedding Table
        self.table = nn.Embedding(
            config.vocab_size,
            config.embed_dim,
            padding_idx=self.pad_id,
        )

        # Positional Encoding
        self.position = PositionalEncoding(config)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initializes token embedding weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.table.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes token + positional embeddings.

        Args:
            x (torch.Tensor): Input token IDs of shape (B, T).

        Returns:
            torch.Tensor: Embedded representation of shape (B, T, C).
        """
        x = x.long()  # Ensure input is of type `long`
        y = self.table(x)  # Token embeddings
        return self.position(y)  # Add positional encoding
