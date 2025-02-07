"""
Module: mini.transformer.encoding
Description: Encoding blocks for transformer models.
"""

from torch import nn, torch

from mini.transformer.config import TransformerConfig


class PositionalEncoding(nn.Module):
    """Positional Encoder for the Transformer model."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embed_dim = config.embed_dim  # (C)
        self.max_seq_len = config.max_seq_len  # (T)
        self.dropout = nn.Dropout(config.dropout)

        # Directly register buffer without creating a duplicate local variable
        self.register_buffer("pe", self._create_positional_encoding().unsqueeze(0))

    def _create_positional_encoding(self) -> torch.Tensor:
        # Create a zero matrix of shape (max_seq_len, embed_dim)
        pe = torch.zeros(self.max_seq_len, self.embed_dim)
        # Create a column vector of positions
        position = torch.arange(self.max_seq_len, dtype=torch.float32).unsqueeze(1)
        # Apply frequency scaling to the frequency terms
        div_term = torch.pow(
            10000.0, torch.arange(0, self.embed_dim, 2).float() / self.embed_dim
        )
        # Compute the positional encoding for sine and cosine terms
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices for sine terms
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices for cosine terms
        return pe  # Return the positional encoding tensor

    def forward(self, x):
        """Ensure output shape remains (B, T, C)"""
        return self.dropout(x + self.pe[:, : x.shape[1], :])


class LinearPositionalEncoding(nn.Module):
    """Linear Positional Encoding for the Transformer model."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.max_seq_len = config.max_seq_len
        self.num_heads = config.num_heads

        # Compute hidden dim dynamically
        self.hidden_dim = self.embed_dim // self.num_heads  # Head dim

        # Fixed Sinusoidal Encoding
        pe = torch.zeros(self.max_seq_len, self.embed_dim)
        position = torch.arange(self.max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2)
            * (-torch.log(torch.tensor(10000.0)) / self.embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # Non-trainable buffer

        # Low-rank projection using `hidden_dim`
        self.projection = nn.Sequential(
            nn.Linear(self.max_seq_len, self.hidden_dim),  # Compress position encoding
            nn.SiLU(),  # Activation
            nn.Linear(self.hidden_dim, self.embed_dim),  # Expand back
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds learned positional encoding to the input tensor."""
        # Project learned positions
        return x + self.projection(self.pe[:, : x.size(1), :])


class RotaryPositionalEncoding(nn.Module):
    """Rotary Positional Encoding for the MiniTransformer model."""

    pass  # To be implemented
