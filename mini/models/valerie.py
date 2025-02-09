"""
File: mini/model.py
Description: Scratchpad for experimenting with transformer concepts.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentencepiece import SentencePieceProcessor


@dataclass
class Parameters:
    # Encoding and Embedding parameters
    vocab_size = 32000
    pad_id = -1
    batch_size = 3
    max_seq_len = 6  # Adjusted to match tokenized input
    seq_len = 4
    embed_dim = 8  # Must be even for sin/cos encoding
    dropout = 0.1
    # Attention parameters
    bias = False
    num_heads = 2

    def __post_init__(self):
        ERROR_MSG = "Embedding dimension must be even for sin/cos encoding"
        assert self.embed_dim % self.num_heads == 0, ERROR_MSG
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5


# Generate sinusoidal encoding
def _generate_sinusoidal_encoding(l_max: int, d_e: int) -> torch.Tensor:
    """Creates sinusoidal positional encodings as described in Vaswani et al. (2017)."""
    assert d_e % 2 == 0, "Embedding dimension must be even for sin/cos encoding"
    pe = torch.zeros(l_max, d_e)
    position = torch.arange(l_max, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_e, 2, dtype=torch.float32)
        * (-torch.log(torch.tensor(10000.0)) / d_e)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, l_max, d_e)


class PositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding for transformer models."""

    def __init__(self, d_e: int, l_max: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", _generate_sinusoidal_encoding(l_max, d_e))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


def _pad_id_mask(input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Create a padding mask from input IDs before they are embedded."""
    return (input_ids == pad_id).unsqueeze(1).unsqueeze(2)  # Shape: [B, 1, 1, T]


def _causal_mask(seqlen: int, device: torch.device) -> torch.Tensor:
    """Create a causal mask to prevent attending to future tokens."""
    mask = torch.full((seqlen, seqlen), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)  # Upper triangular causal mask


def main():
    # Tokenizer setup
    model_file = "models/tokenizer.model"
    tokenizer = SentencePieceProcessor(model_file=model_file)

    # Simulate embedding layer
    embedding = nn.Embedding(vocab_size, embed_dim)
    nn.init.xavier_uniform_(embedding.weight)
    positional_encoding = PositionalEncoding(embed_dim, seq_len, dropout)

    # Dummy input sequences
    sentences = ["The quick brown fox", "jumps over the lazy", "dog"]
    input_ids = [tokenizer.encode(sentence) for sentence in sentences]
    print(
        f"input ids: x={len(input_ids)}, y={max(len(i) for i in input_ids)}, input_ids={input_ids}"
    )

    # Pad or truncate to match seq_len
    input_ids = torch.tensor(
        [ids + [pad_id] * (seq_len - len(ids)) for ids in input_ids]
    )
    print(
        f"input ids: x={len(input_ids)}, y={max(len(i) for i in input_ids)}, input_ids={input_ids}"
    )

    # Apply embedding and positional encoding
    embedded = embedding(input_ids)
    encoded = positional_encoding(embedded)

    # Print shapes for validation
    print(f"Input IDs: {input_ids.shape}")  # Expected: [batch_size, seq_len]
    print(f"Embedded: {embedded.shape}")  # Expected: [batch_size, seq_len, embed_dim]
    print(f"Encoded: {encoded.shape}")  # Expected: [batch_size, seq_len, embed_dim]

    # Create query, key, value, and projection matrices for attention mechanism
    wq = nn.Linear(embed_dim, embed_dim, bias=bias)
    wk = nn.Linear(embed_dim, embed_dim, bias=bias)
    wv = nn.Linear(embed_dim, embed_dim, bias=bias)
    wo = nn.Linear(embed_dim, embed_dim, bias=bias)

    # Initialize weights
    nn.init.xavier_normal_(wq.weight.data)
    nn.init.xavier_normal_(wk.weight.data)
    nn.init.xavier_normal_(wv.weight.data)
    nn.init.xavier_normal_(wo.weight.data)

    # Compute query, key, and value matrices
    B, T, C = encoded.shape

    # Padding and causal masks
    pad_mask = _pad_id_mask(input_ids, pad_id)  # Shape [B, 1, 1, T]
    causal_mask = _causal_mask(seq_len, input_ids.device).unsqueeze(
        0
    )  # Shape [1, T, T]

    print(f"Pad Mask Shape: {pad_mask.shape}")
    print(f"Causal Mask Shape: {causal_mask.shape}")


if __name__ == "__main__":
    main()
