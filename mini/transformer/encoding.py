"""
Copyright © 2023 Austin Berrio
Module: mini.transformer.encoding
Description: Encoding blocks for transformer models.
"""

import torch
from torch import nn

from mini.transformer.config import TransformerConfig


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
    return pe.unsqueeze(0)  # Shape: (1, max_seq_len, embed_dim)


class PositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding for transformer models."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.max_seq_len = config.max_seq_len
        self.dropout = nn.Dropout(config.dropout)

        # Register precomputed positional encodings as a non-trainable buffer
        self.register_buffer(
            "pe", _generate_sinusoidal_encoding(self.max_seq_len, self.embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Ensures output shape remains (B, T, C)"""
        return self.dropout(x + self.pe[:, : x.size(1), :])


class LinearPositionalEncoding(nn.Module):
    """Low-rank learned projection of sinusoidal positional encodings."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.max_seq_len = config.max_seq_len
        self.hidden_dim = self.embed_dim // config.num_heads  # Head dim

        # Register sinusoidal encoding as a non-trainable buffer
        self.register_buffer(
            "pe", _generate_sinusoidal_encoding(self.max_seq_len, self.embed_dim)
        )

        # Projection layers for learnable compression and expansion
        self.projection = nn.Sequential(
            nn.Linear(self.max_seq_len, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds learned positional encoding to the input tensor."""
        projected_pe = self.projection(self.pe[:, : x.size(1), :])
        return x + projected_pe


class RotaryPositionalEncoding(nn.Module):
    """Rotary Positional Encoding for transformer models."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        raise NotImplementedError("RotaryPositionalEncoding is not yet implemented.")
