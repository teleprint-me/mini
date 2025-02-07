"""
Copyright © 2023 Austin Berrio
Module: mini.transformer.embedding
Description: Embedding blocks for transformer models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini.transformer.config import TransformerConfig
from mini.transformer.encoding import LinearPositionalEncoding, PositionalEncoding


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
        self.pad_id = config.pad_id
        self.bias = config.bias

        # We do not need a mask because padding_idx ignores pad_id
        self.tokens = nn.Embedding(
            config.vocab_size,
            config.embed_dim,
            padding_idx=self.pad_id,
        )

        # Positional encoding is added to the token embeddings
        self.positions = LinearPositionalEncoding(config)

        # Hidden layer shapes needs to match positional encoding shape
        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    # 1. Add learnable layers to embedding block
                    nn.Linear(
                        config.embed_dim if i == 0 else config.max_seq_len,
                        config.max_seq_len,
                        device=config.device,
                    ),
                    # 2. Activate learnable positions
                    nn.SiLU(),
                    # 3. Normalize learned positions
                    nn.LayerNorm(config.max_seq_len, device=config.device),
                    # 4. Dropout for regularization
                    nn.Dropout(config.dropout),
                )
                # Save memory for transformer blocks
                for i in range(3)  # We don't want the embedding block to be deep
            ]
        )

        # Projection layer to convert hidden layer output to token embeddings
        self.projection = nn.Linear(config.max_seq_len, config.embed_dim)
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        self.tokens.weight.data.normal_(mean=0.0, std=0.02)
        if self.pad_id > 0:  # Ensure padding embeddings remain zero
            self.tokens.weight.data[self.pad_id] = 0
        for layer in self.hidden_layers:
            nn.init.xavier_uniform_(layer.linear.weight)
            nn.init.xavier_uniform_(layer.norm.weight)
            if self.bias:
                nn.init.uniform_(layer.norm.bias, a=-0.1, b=0.1)
                nn.init.uniform_(layer.linear.bias, a=-0.1, b=0.1)
        nn.init.xavier_uniform_(self.projection.weight)
        if self.bias:
            nn.init.uniform_(self.projection.bias, a=-0.1, b=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape is (B, T)
        x = self.tokens(x)  # Shape becomes (B, T, C)
        x = self.positions(x)  # Encode positions
        for layer in self.hidden_layers:
            x = layer(x)  # Apply each linear layer
        x = self.dropout(self.norm(x))  # Apply normalization & dropout
        return self.projection(x)  # Output is (B, T, C)
