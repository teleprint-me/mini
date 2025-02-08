"""
Copyright © 2023 Austin Berrio
Module: mini.modules.encoding
Description: Encoding blocks for transformer models.
"""

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


class BertEncoding(nn.Module):
    """Bert-style positional encoding."""

    def __init__(self, config: ConfigTransformer):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.max_seq_len = config.max_seq_len
        self.dropout = nn.Dropout(config.dropout)
        # A popular alternative method is used in BERT. arxiv.org/abs/1810.04805
        # Simple element-wise addition with nn.Parameter() is sufficient.
        # Precomputed positional encodings are trainable.
        self.pe = nn.Parameter(
            data=_generate_sinusoidal_encoding(self.max_seq_len, self.embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor x."""
        return self.dropout(x + self.pe[:, : x.size(1), :])


class LinearEncoding(nn.Module):
    """Low-rank learned projection of sinusoidal positional encodings."""

    def __init__(self, config: ConfigTransformer):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.max_seq_len = config.max_seq_len
        # Align with the number of heads to ensure compatibility with multi-head attention.
        self.head_dim = self.embed_dim // config.num_heads

        # Register sinusoidal encoding as a non-trainable buffer
        self.register_buffer(
            "pe", _generate_sinusoidal_encoding(self.max_seq_len, self.embed_dim)
        )

        # Projection layers for learnable compression and expansion
        # NOTE: This is experimental and can be substituted with other methods.
        # TODO: Evaluate the effectiveness of this approach.
        self.projection = nn.Sequential(
            nn.Linear(self.embed_dim, self.head_dim),
            nn.SiLU(),
            nn.Linear(self.head_dim, self.embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds learned positional encoding to the input tensor."""
        projected_pe = self.projection(self.pe[:, : x.size(1), :])
        return x + projected_pe


class RotaryEncoding(nn.Module):
    """Rotary Positional Encoding for transformer models."""

    def __init__(self, config: ConfigTransformer):
        super().__init__()
        raise NotImplementedError("RotaryPositionalEncoding is not yet implemented.")
