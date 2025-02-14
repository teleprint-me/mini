"""
Copyright © 2023 Austin Berrio
Module: mini.modules.embedding
Description: Embedding blocks for transformer models.
"""

import math

import torch
import torch.nn as nn

from mini.config import ConfigTransformer
from mini.modules.encoding import BertEncoding, LinearEncoding, PositionalEncoding
from mini.modules.mlp import MultiLayerPerceptron


class BaseEmbedding(nn.Module):
    """Base class for embedding layers in transformer models."""

    def __init__(self, config: ConfigTransformer):
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(
            config.vocab_size, config.embed_dim, padding_idx=config.pad_id
        )
        self.encodings = None
        self.dropout = nn.Dropout(config.dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initializes token embedding weights using Xavier uniform initialization."""
        # ref: https://github.dev/karpathy/minGPT
        nn.init.normal_(
            self.tokens.weight, std=0.02 / math.sqrt(2 * self.config.num_layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes embeddings for the input tokens.
        Args:
            x (torch.Tensor): Input tokens of shape (batch_size, seq_len).
        Returns:
            torch.Tensor: Embeddings of shape (batch_size, seq_len, embed_dim).
        """
        # Set guard to ensure positional encodings are implemented by subclasses.
        if self.encodings is None:
            raise ValueError("Positional encodings must be implemented by subclasses.")
        # Compute embeddings
        tokens = self.tokens(x)  # Token embeddings
        encodings = self.encodings(tokens)  # Add positional encoding
        return self.dropout(encodings)  # Apply dropout


class PositionalEmbedding(BaseEmbedding):
    """Handles token and positional embeddings for the transformer model."""

    def __init__(self, config: ConfigTransformer):
        super().__init__(config)
        self.encodings = PositionalEncoding(config)


class BertEmbedding(BaseEmbedding):
    """Handles token and positional embeddings for the transformer model."""

    def __init__(self, config: ConfigTransformer):
        super().__init__(config)
        self.encodings = BertEncoding(config)


# Projection layers for learnable compression and expansion
# NOTE: This is experimental and can be substituted with other methods.
# WARN: Training will oscillate if not properly tuned and model will be trapped in a local minimum.
# This is due to the fact that the positional encodings are distorted by the learned projection.
# TODO: Implement a reward and penalty mechanism to stabilize training.
class LinearEmbedding(BaseEmbedding):
    """Handles token and positional embeddings for the transformer model with an MLP projection."""

    def __init__(self, config):
        super().__init__(config)
        # Learnable positional encoding
        self.encodings = LinearEncoding(config)
        # LayerNorm to stabilize positional encodings
        self.norm = nn.LayerNorm(config.embed_dim)
        # MLP learns **positional encodings** (avoids distorting tokens)
        self.mlp = MultiLayerPerceptron(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes token + positional embeddings and applies MLP layers."""
        tokens = self.tokens(x)  # Token embeddings (B, T, C)
        encodings = self.norm(self.encodings(tokens))  # Normalize positional encodings
        return tokens + self.mlp(encodings)  # Residual-style learning for stability
