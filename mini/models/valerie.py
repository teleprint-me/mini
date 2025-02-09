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
    batch_size: int = 3  # Batch size
    max_seq_len: int = 8  # Adjusted to match tokenized input
    seq_len: int = 6  # Sequence length
    embed_dim: int = 10  # Must be even for sin/cos encoding
    dropout: float = 0.1  # Dropout rate
    bias: bool = False  #
    num_heads: int = 2  # Number of attention heads

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

    def __init__(self, d_e: int, l_max: int, p: float = 0.1):
        super().__init__()
        self.register_buffer("pe", _generate_sinusoidal_encoding(l_max, d_e))
        self.dropout = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


class PositionalEmbedding(nn.Module):
    def __init__(self, V: int, d_e: int, l_max: int, p: float = 0.1):
        super().__init__()
        self.tokens = nn.Embedding(V, d_e)
        nn.init.xavier_uniform_(self.tokens.weight)
        self.positions = PositionalEncoding(d_e, l_max, p)
        self.dropout = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected: [B, T]
        print(f"PositionalEmbedding Input: {x.shape}")
        x = self.tokens(x)  # Shape: [B, T, d_e]
        # Expected: [batch_size, seq_len, embed_dim]
        print(f"PositionalEmbedding.tokens: {x.shape}")
        x = self.positions(x)  # Shape: [B, T, d_e]
        # Expected: [batch_size, seq_len, embed_dim]
        print(f"PositionalEmbedding.positions: {x.shape}")
        return self.dropout(x)


class SequenceMask:
    @staticmethod
    def pad_id_mask(input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
        """Create a padding mask from input IDs before they are embedded."""
        return (input_ids == pad_id).unsqueeze(1).unsqueeze(2)  # Shape: [B, 1, 1, T]

    @staticmethod
    def causal_mask(seqlen: int, device: torch.device) -> torch.Tensor:
        """Create a causal mask to prevent attending to future tokens."""
        mask = torch.full((seqlen, seqlen), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)  # Upper triangular causal mask

    @staticmethod
    def combined_mask(
        pad_mask: torch.Tensor, causal_mask: torch.Tensor
    ) -> torch.Tensor:
        """Combine padding and causal masks."""
        return pad_mask + causal_mask


class CausalAttention(nn.Module):
    def __init__(self, d_k: int, d_v: int, d_e: int, dropout: float = 0.1):
        super().__init__()


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
    embedding = PositionalEmbedding(
        V=params.vocab_size,
        d_e=params.embed_dim,
        l_max=params.max_seq_len,
        p=params.dropout,
    )

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
    print(f"Embedded: {d_model.shape}")

    # Create query, key, value, and projection matrices for attention mechanism
    wq = nn.Linear(params.embed_dim, params.embed_dim, bias=params.bias)
    wk = nn.Linear(params.embed_dim, params.embed_dim, bias=params.bias)
    wv = nn.Linear(params.embed_dim, params.embed_dim, bias=params.bias)
    wo = nn.Linear(params.embed_dim, params.embed_dim, bias=params.bias)

    # Initialize weights
    nn.init.xavier_normal_(wq.weight.data)
    nn.init.xavier_normal_(wk.weight.data)
    nn.init.xavier_normal_(wv.weight.data)
    nn.init.xavier_normal_(wo.weight.data)

    # Compute query, key, and value matrices
    B, T, C = d_model.shape
    # Compute Q, K, V matrices
    q, k, v = wq(d_model), wk(d_model), wv(d_model)
    # Reshape to multiple heads: [B, T, H, D] -> [B, H, T, D]
    q, k, v = [
        x.view(B, T, params.num_heads, params.head_dim).transpose(1, 2)
        for x in [q, k, v]
    ]
    # Compute scaled dot product attention [B, H, T, T]
    attn_weights = (q @ k.transpose(-2, -1)) * params.scale

    # Padding mask, Shape [B, 1, 1, T]
    pad_mask = SequenceMask.pad_id_mask(input_ids, params.pad_id)

    # Causal mask, Shape [1, T, T]
    causal_mask = SequenceMask.causal_mask(
        params.max_seq_len, input_ids.device
    ).unsqueeze(0)
    # Combine padding and causal mask
    combined_mask = pad_mask + causal_mask
    if combined_mask is not None:
        attn_weights = attn_weights.masked_fill(
            combined_mask == float("-inf"), float("-inf")
        )

    print(f"Pad Mask Shape: {pad_mask.shape}")
    print(f"Causal Mask Shape: {causal_mask.shape}")
    print(f"Combined Mask Shape: {combined_mask.shape}")

    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, C)
    print(f"Attention Output Shape: {attn_out.shape}")


if __name__ == "__main__":
    main()
