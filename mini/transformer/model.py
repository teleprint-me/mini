"""
Copyright © 2023 Austin Berrio
Module: mini.transformer.model
Description: A simplified transformer model for baseline training.
"""

import functools
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MiniRuntime:
    """Manages runtime-specific settings like device handling and seeding."""

    seed: int = 42

    @functools.cached_property
    def device_name(self) -> str:
        """Returns the best available device name ('cuda' or 'cpu')."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            return "cuda"
        return "cpu"

    @functools.cached_property
    def device_type(self) -> torch.device:
        """Returns the best available device as a `torch.device` object."""
        return torch.device(self.device_name)

    def seed_all(self) -> None:
        """Sets the random seed for reproducibility."""
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)


@dataclass
class MiniConfig:
    vocab_size: int = 32000
    embed_dim: int = 512
    num_heads: int = 8
    head_dim: int = 32
    num_layers: int = 8
    ff_dim: int = 256
    max_seq_len: int = 512
    pad_id: int = -1
    dropout: float = 0.1
    eps: float = 1e-8
    theta: float = 10000.0
    bias: bool = False

    def as_dict(self) -> dict[str, any]:
        """Returns a dictionary representation of the config."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class RMSNorm(nn.Module):
    def __init__(self, config: MiniConfig):
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.ones(config.embed_dim))  # Gamma parameter

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * self._norm(x.float()).type_as(x)


class FeedForward(nn.Module):
    def __init__(self, config: MiniConfig):
        super().__init__()
        self.bias = config.bias

        self.w1 = nn.Linear(config.embed_dim, config.ff_dim, bias=config.bias)
        self.w2 = nn.Linear(config.ff_dim, config.embed_dim, bias=config.bias)
        self.w3 = nn.Linear(config.embed_dim, config.ff_dim, bias=config.bias)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)
        if self.bias:
            nn.init.uniform_(self.w1.bias)
            nn.init.uniform_(self.w2.bias)
            nn.init.uniform_(self.w3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MiniAttention(nn.Module):
    def __init__(self, config: MiniConfig):
        super().__init__()
        self.pad_id = max(config.pad_id, 0)
        self.bias = config.bias
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.scale = self.head_dim**-0.5

        self.wq = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        self.wk = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        self.wv = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        self.wo = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.wq.weight)
        nn.init.xavier_uniform_(self.wk.weight)
        nn.init.xavier_uniform_(self.wv.weight)
        nn.init.xavier_uniform_(self.wo.weight)
        if self.bias:
            nn.init.uniform_(self.wq.bias)
            nn.init.uniform_(self.wk.bias)
            nn.init.uniform_(self.wv.bias)
            nn.init.uniform_(self.wo.bias)

    def _mask(self, x: torch.Tensor) -> torch.Tensor:
        """Creates an attention mask of shape [B, 1, T, T] to ignore pad tokens."""
        B, T, _ = x.shape  # Extract batch size and sequence length
        mask = (x[:, :, 0] != self.pad_id).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
        return mask.expand(B, 1, T, T)  # Expand to [B, 1, T, T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch size, sequence length, embedding dimension
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q, k, v = [
            t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            for t in (q, k, v)
        ]

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, T, T]

        # Compute and apply mask
        mask = self._mask(x)  # Ensure shape [B, 1, T, T]
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)


class MiniBlock(nn.Module):
    def __init__(self, config: MiniConfig):
        super().__init__()
        self.attn = MiniAttention(config)
        self.norm1 = RMSNorm(config)
        self.norm2 = RMSNorm(config)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x


class PositionalEncoding(nn.Module):
    """Positional Encoder for the MiniTransformer model."""

    def __init__(self, config: MiniConfig):
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


class MiniEmbedding(nn.Module):
    """Handles token and positional embeddings for the transformer model."""

    def __init__(self, config: MiniConfig):
        super().__init__()
        self.bias = config.bias

        # We do not need a mask because padding_idx ignores pad_id
        self.tokens = nn.Embedding(
            config.vocab_size,
            config.embed_dim,
            padding_idx=self.pad_id,
        )
        self.positions = PositionalEncoding(config)

        # Hidden layer shape needs to match positional encoding shape
        self.hidden = nn.Linear(config.embed_dim, config.max_seq_len)
        self.projection = nn.Linear(config.max_seq_len, config.embed_dim)

        self.norm = nn.LayerNorm(config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)

        self._init_weights()

    def _init_weights(self):
        self.tokens.weight.data.normal_(mean=0.0, std=0.02)
        if self.pad_id > 0:  # Ensure padding embeddings remain zero
            self.tokens.weight.data[self.pad_id] = 0
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.xavier_uniform_(self.projection.weight)
        if self.bias:
            nn.init.uniform_(self.hidden.bias)
            nn.init.uniform_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape is (B, T)
        x = self.tokens(x)  # Shape becomes (B, T, C)
        x = self.positions(x)  # Encode positions
        x = F.silu(self.hidden(x))  # Apply SiLU activation
        x = self.dropout(self.norm(x))  # Apply normalization & dropout
        return self.projection(x)  # Output is (B, T, C)


class MiniTransformer(nn.Module):
    def __init__(self, config: MiniConfig):
        super().__init__()
        self.bias = config.bias

        self.embedding = MiniEmbedding(config)
        self.blocks = nn.ModuleList(
            [MiniBlock(config) for _ in range(config.num_layers)]
        )
        self.norm = RMSNorm(config)
        self.head = nn.Linear(config.embed_dim, config.vocab_size)

        self.max_seq_len = config.max_seq_len
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.head.weight)
        if self.bias is not None:
            nn.init.uniform_(self.head.bias)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)
