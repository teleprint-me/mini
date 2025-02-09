"""
File: mini.models.valerie
Description: Scratchpad for experimenting with transformer concepts.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentencepiece import SentencePieceProcessor


@dataclass
class Parameters:
    vocab_size: int = 32000  # V
    pad_id: int = -1  # -1 is reserved for padding
    max_seq_len: int = 8  # Adjusted to match tokenized input
    seq_len: int = 6  # Sequence length
    embed_dim: int = 10  # Must be even for sin/cos encoding
    dropout: float = 0.1  # Dropout rate
    bias: bool = False  # Learn an additive bias
    num_heads: int = 2  # Number of attention heads
    dtype: torch.dtype = None  # Tensor data type
    device: torch.device = None  # Model device

    def __post_init__(self):
        ERROR_MSG = "Embedding dimension must be even for sin/cos encoding"
        assert self.embed_dim % self.num_heads == 0, ERROR_MSG
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5


# Generate sinusoidal encoding
def _generate_sinusoidal_encoding(params: Parameters) -> torch.Tensor:
    """Creates sinusoidal positional encodings as described in Vaswani et al. (2017)."""
    ERROR_MSG = "Embedding dimension must be even for sin/cos encoding"
    assert params.embed_dim % 2 == 0, ERROR_MSG
    pe = torch.zeros(params.max_seq_len, params.embed_dim)
    position = torch.arange(params.max_seq_len, dtype=params.dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, params.embed_dim, 2, dtype=params.dtype)
        * (-torch.log(torch.tensor(10000.0)) / params.embed_dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, l_max, d_e)


class PositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding for transformer models."""

    def __init__(self, params: Parameters):
        super().__init__()
        frequency = _generate_sinusoidal_encoding(params)
        self.register_buffer("frequency", frequency)
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        max_seq_len = tokens.size(1)
        return self.dropout(tokens + self.frequency[:, :max_seq_len, :])


class PositionalEmbedding(nn.Module):
    def __init__(self, params: Parameters):
        super().__init__()

        # Initialize the token embedding layer
        self.tokens = nn.Embedding(params.vocab_size, params.embed_dim)
        nn.init.xavier_uniform_(self.tokens.weight)

        # Initialize the positional encoding layer
        self.encodings = PositionalEncoding(params)

        # Initialize the dropout layer
        self.dropout_layer = nn.Dropout(params.dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the PositionalEmbedding module.

        Args:
            input_ids (torch.Tensor): Input tensor of shape [batch_size, max_seq_len]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, max_seq_len, embed_dim]
        """
        # Embed the input tokens into token embeddings
        tokens = self.tokens(input_ids)
        # Shape: [batch_size, max_seq_len, embed_dim]
        print(f"Token embeddings shape: {tokens.shape}")

        # Apply positional encoding to the token embeddings
        encodings = self.encodings(tokens)
        # Shape: [batch_size, max_seq_len, embed_dim]
        print(f"Positional encodings shape: {encodings.shape}")

        # Apply dropout to the combined embeddings
        d_model = self.dropout_layer(encodings)
        # Shape: [batch_size, max_seq_len, embed_dim]
        print(f"Output embeddings shape: {d_model.shape}")

        return d_model


class SequenceMask:
    def __init__(self, params: Parameters):
        self.pad_id = params.pad_id
        self.max_seq_len = params.max_seq_len
        self.device = params.device

    def __call__(
        self, input_ids: torch.Tensor, mask_type: str = "causal"
    ) -> torch.Tensor:
        """Combine padding and causal masks."""
        pad_mask = self._pad_mask(input_ids)
        if mask_type == "causal":
            mask = self._causal_mask()
        elif mask_type == "bidirectional":
            mask = self._bidirectional_mask()
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")
        return pad_mask + mask  # Add masks together

    def _pad_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create a padding mask of shape [B, 1, 1, T] from input IDs before they are embedded."""
        return (input_ids == self.pad_id).unsqueeze(1).unsqueeze(2)

    def _bidirectional_mask(self) -> torch.Tensor:
        """Create a bidirectional mask of shape [1, T, T] to prevent attending to future tokens."""
        # Square matrix for bidirectional mask
        size = (self.max_seq_len, self.max_seq_len)
        mask = torch.full(size, float("-inf"), device=self.device)
        # Upper triangular bidirectional mask
        return torch.triu(mask, diagonal=1)

    def _causal_mask(self) -> torch.Tensor:
        """Create a causal mask of shape [1, T, T] to prevent attending to future tokens."""
        # Square matrix for causal mask
        size = (self.max_seq_len, self.max_seq_len)
        mask = torch.full(size, float("-inf"), device=self.device)
        # Upper triangular causal mask
        return torch.triu(mask, diagonal=1)


class CausalAttention(nn.Module):
    def __init__(self, params: Parameters):
        super().__init__()
        self.num_heads = params.num_heads
        self.head_dim = params.head_dim
        self.scale = params.scale

        # Linear layers for Q, K, V
        self.wq = nn.Linear(params.embed_dim, params.embed_dim, bias=params.bias)
        self.wk = nn.Linear(params.embed_dim, params.embed_dim, bias=params.bias)
        self.wv = nn.Linear(params.embed_dim, params.embed_dim, bias=params.bias)

        # Output projection
        self.wo = nn.Linear(params.embed_dim, params.embed_dim, bias=params.bias)

        # Initialize weights
        nn.init.xavier_normal_(self.wq.weight)
        nn.init.xavier_normal_(self.wk.weight)
        nn.init.xavier_normal_(self.wv.weight)
        nn.init.xavier_normal_(self.wo.weight)

    def forward(self, d_in: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, C = d_in.shape

        # Compute Q, K, V projections
        q, k, v = self.wq(d_in), self.wk(d_in), self.wv(d_in)

        # Reshape to multiple heads: [B, T, d_model] → [B, num_heads, T, head_dim]
        q, k, v = [
            t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            for t in (q, k, v)
        ]

        # Compute scaled dot-product attention
        d_attn = (q @ k.transpose(-2, -1)) * self.scale
        d_attn = d_attn.masked_fill(mask == float("-inf"), float("-inf"))
        d_attn = F.softmax(d_attn, dim=-1)

        # Apply multi-head attention weights to values
        d_out = d_attn @ v  # [B, num_heads, T, head_dim]

        # Reshape: concatenate heads → [B, T, d_model]
        d_out = d_out.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear projection
        return self.wo(d_out)


def main():
    # Tokenizer setup
    model_file = "models/tokenizer.model"
    tokenizer = SentencePieceProcessor(model_file=model_file)
    params = Parameters(
        vocab_size=tokenizer.vocab_size(),
        pad_id=max(tokenizer.pad_id(), 0),
    )
    print(f"Parameters: {params}")

    # Simulate embedding layer
    embedding = PositionalEmbedding(params)

    # Dummy input sequences
    sentences = ["The quick brown fox", "jumps over the lazy", "dog"]
    input_ids = [tokenizer.encode(sentence) for sentence in sentences]

    # Pad or truncate to match seq_len
    input_ids = torch.tensor(
        [ids + [params.pad_id] * (params.max_seq_len - len(ids)) for ids in input_ids]
    )
    # Expected: [batch_size, seq_len]
    print(f"Input IDs: {input_ids.shape}")

    # Apply embedding and positional encoding
    d_model = embedding(input_ids)

    # Expected: [batch_size, seq_len, embed_dim]
    print(f"d_model: {d_model.shape}")

    # Padding mask, Shape [B, 1, 1, T]
    sequence_mask = SequenceMask(params=params)
    mask = sequence_mask(input_ids)  # Pad mask, Shape [B, 1, 1, T]
    print(f"Mask Shape: {mask.shape}")  # Causal mask, Shape [B, 1, T, T]

    # Apply attention mechanism
    attention = CausalAttention(params=params)
    projection = attention(d_model, mask=mask)
    print(f"Attention projection Shape: {projection.shape}")  # [B, seq_len, embed_dim]


if __name__ == "__main__":
    main()
