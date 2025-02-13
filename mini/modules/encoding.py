"""
Copyright © 2023 Austin Berrio
Module: mini.modules.encoding
Description: Encoding blocks for transformer models.
"""

from typing import Tuple

import torch
from torch import nn

from mini.config import ConfigTransformer


# Self-Attention with Relative Position Representations
# https://arxiv.org/abs/1803.02155
class BaseEncoding(nn.Module):
    """Base class for encoding blocks."""

    def __init__(self, config: ConfigTransformer):
        super().__init__()
        self.config = config
        self.embed_dim = self.config.embed_dim
        self.max_seq_len = self.config.max_seq_len
        self.dtype = self.config.dtype
        self.dropout = nn.Dropout(self.config.dropout)

    # NOTE: Not sure if this is mathematically sound. Need to revisit this if issues arise.
    def _generate_sinusoidal_encoding(self) -> torch.Tensor:
        """Creates sinusoidal positional encodings as described in Vaswani et al. (2017)."""
        # Create a tensor of shape (max_seq_len, embed_dim) to store the positional encodings.
        pe = torch.zeros(self.max_seq_len, self.embed_dim)
        # Create a tensor of shape (max_seq_len, 1) to store the position indices.
        position = torch.arange(self.max_seq_len, dtype=self.dtype).unsqueeze(1)
        # Calculate the division term for the sinusoidal encoding of shape (D/2,) to avoid division by zero.
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2, dtype=self.dtype)
            * (-torch.log(torch.tensor(10000.0)) / self.embed_dim)
        )
        # Compute the positional encoding for even and odd indices separately.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor x."""
        raise NotImplementedError("Forward method must be implemented by subclasses.")


# Self-Attention with Relative Position Representations
# https://arxiv.org/abs/1803.02155
class PositionalEncoding(BaseEncoding):
    """Standard fixed sinusoidal positional encoding for transformer models."""

    def __init__(self, config: ConfigTransformer):
        super().__init__(config)
        # Register precomputed positional encodings as a non-trainable buffer
        self.register_buffer("pe", self._generate_sinusoidal_encoding())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor x."""
        seq_len = x.size(1)
        return self.dropout(x + self.pe[:, :seq_len, :])


# BERT: Pre-training of Bidirectional Encoder Representations from Transformers
# https://arxiv.org/abs/1810.04805
class BertEncoding(BaseEncoding):
    """Bert-style positional encoding."""

    def __init__(self, config: ConfigTransformer):
        super().__init__(config)
        # Simple element-wise addition with nn.Parameter() for trainable parameters is sufficient.
        self.pe = nn.Parameter(data=self._generate_sinusoidal_encoding())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor x."""
        seq_len = x.size(1)
        return self.dropout(x + self.pe[:, :seq_len, :])


# Projection layers for learnable compression and expansion
# NOTE: This is experimental and can be substituted with other methods.
# WARN: Training will oscillate if not properly tuned and model will be trapped in a local minimum.
# This is due to the fact that the positional encodings are distorted by the learned projection.
class LinearEncoding(BaseEncoding):
    """Low-rank learned projection of sinusoidal positional encodings."""

    def __init__(self, config: ConfigTransformer):
        super().__init__(config)
        # Register sinusoidal encoding as a non-trainable buffer
        self.register_buffer("pe", self._generate_sinusoidal_encoding())

        # TODO: Evaluate the effectiveness of this approach.
        self.projection = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds learned positional encoding to the input tensor."""
        seq_len = x.size(1)
        projected_pe = self.projection(self.pe[:, :seq_len, :])
        return self.dropout(projected_pe) + x


# NOTE: This must be applied to the query and key tensors before the attention mechanism.
# RoFormer: Enhanced Transformer with Rotary Position Embeddings
# https://arxiv.org/abs/2104.09864
class RotaryEncoding(nn.Module):
    """Rotary Positional Encoding (RoPE) for transformer models."""

    def __init__(self, config: ConfigTransformer):
        super().__init__()
        # Precompute the rotation angles for RoPE
        theta = config.rope_theta ** (
            -torch.arange(0, config.embed_dim, 2, dtype=config.dtype) / config.embed_dim
        )
        position = torch.arange(config.max_seq_len, dtype=config.dtype).unsqueeze(1)
        angles = position * theta.unsqueeze(0)
        self.register_buffer("cos", torch.cos(angles))
        self.register_buffer("sin", torch.sin(angles))

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates tensor according to RoPE formulation."""
        seq_len = x.size(1)
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        x_rotated_even = x_even * self.cos[:seq_len, :] - x_odd * self.sin[:seq_len, :]
        x_rotated_odd = x_odd * self.cos[:seq_len, :] + x_even * self.sin[:seq_len, :]
        return torch.stack([x_rotated_even, x_rotated_odd], dim=-1).flatten(-2)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors."""
        return self.rotate(q), self.rotate(k)
