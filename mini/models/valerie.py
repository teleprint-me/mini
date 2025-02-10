"""
File: mini.models.valerie
Description: Scratchpad for experimenting with transformer concepts.
"""

import random
from dataclasses import dataclass, field

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
    # Tensor data type for computations
    dtype: torch.dtype = field(init=False, default=torch.float32)
    # Device name for computations
    dname: str = field(init=False, default="cpu")
    seed: int = 42  # Random seed for reproducibility

    def __post_init__(self):
        """Initializes the model parameters and sets the device for computations."""
        # Set the pad id to the initial index if it is not set
        self.pad_id = max(self.pad_id, 0)
        # Set the head dimension and scale for attention
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        # Assert that the model parameters are initialized correctly
        self.__assert_init()
        # Set the device for computations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.dname = "cuda"

    def __assert_init(self):
        """Asserts that the model parameters are initialized correctly."""
        ERROR_MSG = "Pad id must be 0 or greater"
        assert self.pad_id >= 0, ERROR_MSG
        ERROR_MSG = "Embedding dimension must be even for sin/cos encoding"
        assert self.embed_dim % 2 == 0, ERROR_MSG
        ERROR_MSG = "Embedding dimension must be divisible by the number of heads"
        assert self.embed_dim % self.num_heads == 0, ERROR_MSG
        ERROR_MSG = "Head dimension must be equal to embedding dimension divided by the number of heads"
        assert self.head_dim == self.embed_dim // self.num_heads, ERROR_MSG
        ERROR_MSG = "Scale must be equal to the square root of the head dimension"
        assert self.scale == self.head_dim**-0.5, ERROR_MSG

    @property
    def device(self) -> torch.device:
        """Returns the best available device as a `torch.device` object."""
        return torch.device(self.dname)

    def set_seed(self) -> None:
        """Sets the random seed for reproducibility."""
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)


# Generate sinusoidal encoding
def _generate_sinusoidal_encoding(params: Parameters) -> torch.Tensor:
    """Creates sinusoidal positional encodings as described in Vaswani et al. (2017)."""
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
        # Shape: [batch_size, max_seq_len, embed_dim]
        tokens = self.tokens(input_ids)

        # Apply positional encoding to the token embeddings
        # Shape: [batch_size, max_seq_len, embed_dim]
        encodings = self.encodings(tokens)

        # Apply dropout to the combined embeddings
        # Shape: [batch_size, max_seq_len, embed_dim]
        return self.dropout_layer(encodings)  # d_model


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
        mask = self._get_mask(mask_type)
        return pad_mask + mask  # Add masks together

    def _pad_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create a padding mask of shape [B, 1, 1, T] from input IDs before they are embedded."""
        return (input_ids == self.pad_id).unsqueeze(1).unsqueeze(2)

    def _get_mask(self, mask_type: str) -> torch.Tensor:
        """Retrieve the appropriate attention mask."""
        if mask_type == "causal":
            return self._causal_mask()
        elif mask_type == "bidirectional":
            return self._bidirectional_mask()
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")

    def _bidirectional_mask(self) -> torch.Tensor:
        """Create a bidirectional mask that only ignores pad tokens."""
        # No upper triangular mask, only ignore pad tokens
        return torch.ones((self.max_seq_len, self.max_seq_len), device=self.device)

    def _causal_mask(self) -> torch.Tensor:
        """Create a causal mask to prevent attending to future tokens."""
        size = (self.max_seq_len, self.max_seq_len)  # Square matrix of shape [T, T]
        mask = torch.full(size, float("-inf"), device=self.device)  # Add -inf values
        return torch.triu(mask, diagonal=1)  # Upper triangular matrix of shape [T, T]


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


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network (FFN) used in transformer layers."""

    def __init__(self, params: Parameters):
        super().__init__()
        self.fc1 = nn.Linear(params.embed_dim, params.embed_dim * 4)
        self.fc2 = nn.Linear(params.embed_dim * 4, params.embed_dim)
        self.activation = F.gelu  # Common choice (SiLU also works)
        self.dropout = nn.Dropout(params.dropout)

        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.activation(self.fc1(x))))


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
    embedding = PositionalEmbedding(params).to(params.device)

    # Dummy input sequences
    sentences = ["The quick brown fox", "jumps over the lazy", "dog"]
    input_ids = [tokenizer.encode(sentence) for sentence in sentences]

    # Pad or truncate to match seq_len
    input_ids = torch.tensor(
        [ids + [params.pad_id] * (params.max_seq_len - len(ids)) for ids in input_ids]
    ).to(params.device)
    # Expected: [batch_size, seq_len]
    print(f"Input IDs: {input_ids.shape}")

    # Apply embedding and positional encoding
    d_model = embedding(input_ids)

    # Expected: [batch_size, seq_len, embed_dim]
    print(f"d_model: {d_model.shape}")

    # Padding mask, Shape [B, 1, 1, T], Causal mask, Shape [B, 1, T, T]
    sequence_mask = SequenceMask(params=params)
    # Expected: [batch_size, 1, seq_len, seq_len] for causal mask
    mask = sequence_mask(input_ids, mask_type="causal")

    # Apply attention mechanism
    attention = CausalAttention(params=params).to(params.device)
    projection = attention(d_model, mask=mask)
    print(f"Attention projection Shape: {projection.shape}")  # [B, seq_len, embed_dim]


if __name__ == "__main__":
    main()
