"""
Copyright © 2023 Austin Berrio
Module: mini.modules.encoding
Description: Encoding blocks for transformer models.
"""

from typing import Tuple

import torch
from torch import nn

from mini.config import ConfigTransformer


def _generate_sinusoidal_encoding(max_seq_len: int, embed_dim: int) -> torch.Tensor:
    """Creates sinusoidal positional encodings as described in Vaswani et al. (2017)."""
    pe = torch.zeros(max_seq_len, embed_dim)
    position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float32)
        * (-torch.log(torch.tensor(10000.0)) / embed_dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    # NOTE: Ensure that PE is of shape (batch_size, max_seq_len, embed_dim)
    # This must match the input shape of the attention mechanism.
    return pe.unsqueeze(0)  # (b, l_max, d_e)


# arxiv: arXiv:1803.02155
class PositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding for transformer models."""

    def __init__(self, config: ConfigTransformer):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.max_seq_len = config.max_seq_len
        self.dropout = nn.Dropout(config.dropout)

        # Register precomputed positional encodings as a non-trainable buffer
        self.register_buffer(
            "pe", _generate_sinusoidal_encoding(self.max_seq_len, self.embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor x."""
        return self.dropout(x + self.pe[:, : x.size(1), :])


# arxiv:1810.04805
class BertEncoding(nn.Module):
    """Bert-style positional encoding."""

    def __init__(self, config: ConfigTransformer):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.max_seq_len = config.max_seq_len
        self.dropout = nn.Dropout(config.dropout)
        # Simple element-wise addition with nn.Parameter() for trainable parameters is sufficient.
        self.pe = nn.Parameter(
            data=_generate_sinusoidal_encoding(self.max_seq_len, self.embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor x."""
        return self.dropout(x + self.pe[:, : x.size(1), :])


# Projection layers for learnable compression and expansion
# NOTE: This is experimental and can be substituted with other methods.
# WARN: Training will oscillate if not properly tuned and model will be trapped in a local minimum.
# This is due to the fact that the positional encodings are distorted by the learned projection.
class LinearEncoding(nn.Module):
    """Low-rank learned projection of sinusoidal positional encodings."""

    def __init__(self, config: ConfigTransformer):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.max_seq_len = config.max_seq_len
        # Align with the number of heads to ensure compatibility with multi-head attention.
        self.head_dim = self.embed_dim // config.num_heads
        self.dropout = nn.Dropout(config.dropout)

        # Register sinusoidal encoding as a non-trainable buffer
        self.register_buffer(
            "pe", _generate_sinusoidal_encoding(self.max_seq_len, self.embed_dim)
        )

        # TODO: Evaluate the effectiveness of this approach.
        self.projection = nn.Sequential(
            nn.Linear(self.embed_dim, self.head_dim),
            nn.SiLU(),
            nn.Linear(self.head_dim, self.embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds learned positional encoding to the input tensor."""
        projected_pe = self.projection(self.pe[:, : x.size(1), :])
        return self.dropout(projected_pe) + x


# arXiv:2104.09864
# NOTE: This must be applied to the query and key tensors before the attention mechanism.
class RotaryEncoding(nn.Module):
    """Rotary Positional Encoding (RoPE) for transformer models."""

    def __init__(self, config: ConfigTransformer):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.max_seq_len = config.max_seq_len
        assert self.embed_dim % 2 == 0, "Embedding dimension must be even for RoPE."

        # Precompute the rotation angles for RoPE
        theta = config.rope_theta ** (
            -torch.arange(0, self.embed_dim, 2).float() / self.embed_dim
        )
        position = torch.arange(self.max_seq_len, dtype=torch.float32).unsqueeze(1)
        angles = position * theta.unsqueeze(0)
        self.register_buffer("cos", torch.cos(angles))
        self.register_buffer("sin", torch.sin(angles))

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates tensor according to RoPE formulation."""
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        x_rotated_even = (
            x_even * self.cos[: x.size(1), :] - x_odd * self.sin[: x.size(1), :]
        )
        x_rotated_odd = (
            x_odd * self.cos[: x.size(1), :] + x_even * self.sin[: x.size(1), :]
        )
        return torch.stack([x_rotated_even, x_rotated_odd], dim=-1).flatten(-2)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors."""
        return self.rotate(q), self.rotate(k)
