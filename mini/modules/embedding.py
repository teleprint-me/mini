"""
Copyright © 2023 Austin Berrio
Module: mini.blocks.embedding
Description: Embedding blocks for transformer models.
"""

import torch
import torch.nn as nn

from mini.blocks.encoding import LinearPositionalEncoding, PositionalEncoding
from mini.transformer.config import TransformerConfig


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
        y = self.table(x)  # Token embeddings
        return self.position(y)  # Add positional encoding


class LinearEmbedding(nn.Module):
    """Handles token and positional embeddings for the transformer model."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.pad_id = max(config.pad_id, 0)
        self.bias = config.bias
        self.embed_dim = config.embed_dim

        # Token embedding table
        self.tokens = nn.Embedding(
            config.vocab_size,
            config.embed_dim,
            padding_idx=self.pad_id,
        )

        # Positional encoding (learnable variant)
        self.positions = LinearPositionalEncoding(config)

        # Define MLP layers for embedding transformation
        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        config.embed_dim if i == 0 else config.max_seq_len,
                        config.max_seq_len,
                    ),
                    nn.SiLU(),
                    nn.LayerNorm(config.max_seq_len),
                    nn.Dropout(config.dropout),
                )
                for i in range(config.num_embed_layers)  # Keeping depth small
            ]
        )

        # Final projection to match token embedding space
        self.projection = nn.Linear(config.max_seq_len, config.embed_dim)

        # Dropout & LayerNorm for regularization
        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes all weights using Xavier uniform and bias adjustments."""
        nn.init.xavier_uniform_(self.tokens.weight)
        if self.pad_id > 0:
            self.tokens.weight.data[self.pad_id] = 0  # Ensure padding remains zero

        for layer in self.hidden_layers:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if self.bias:
                        nn.init.uniform_(module.bias, a=-0.1, b=0.1)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.xavier_uniform_(module.weight)
                    if self.bias:
                        nn.init.uniform_(module.bias, a=-0.1, b=0.1)

        nn.init.xavier_uniform_(self.projection.weight)
        if self.bias:
            nn.init.uniform_(self.projection.bias, a=-0.1, b=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes token + positional embeddings and applies MLP layers."""
        x = self.tokens(x)  # (B, T, C)
        x = self.positions(x)  # Add positional encodings

        for layer in self.hidden_layers:
            x = layer(x)  # Apply MLP layers

        x = self.norm(self.dropout(x))  # Regularization & normalization
        return self.projection(x)  # (B, T, C) output


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings for transformers."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        raise NotImplementedError("RotaryEmbedding is not yet implemented.")
