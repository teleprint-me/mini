"""
Copyright © 2023 Austin Berrio
Module: mini.transformer.model
Description: A simplified transformer model for baseline training.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.bias = config.bias
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.scale = self.head_dim**-0.5

        self.wq = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        self.wk = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        self.wv = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        self.wo = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.wq.weight)
        nn.init.xavier_uniform_(self.wk.weight)
        nn.init.xavier_uniform_(self.wv.weight)
        nn.init.xavier_uniform_(self.wo.weight)
        if self.bias:
            nn.init.uniform_(self.wq.bias)
            nn.init.uniform_(self.wk.bias)
            nn.init.uniform_(self.wv.bias)
            nn.init.uniform_(self.wo.bias)

    def forward(self, x, mask=None):
        B, T, C = x.shape  # batch size, sequence length, embedding dimension
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q, k, v = [
            t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            for t in (q, k, v)
        ]
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
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

    def forward(self, x, mask=None):
        x = self.norm1(x + self.attn(x, mask))
        x = self.norm2(x + self.ff(x))
        return x


class MiniEncoder(nn.Module):
    def __init__(self, config: MiniConfig):
        super().__init__()
        self.bias = config.bias
        padding_idx = max(config.pad_id, 0)  # Ensure pad_id is non-negative
        self.embedding = nn.Embedding(
            config.vocab_size, config.embed_dim, padding_idx=padding_idx
        )
        self.hidden = nn.Linear(config.embed_dim, config.max_seq_len)
        self.projection = nn.Linear(config.max_seq_len, config.embed_dim)

        self.norm = nn.LayerNorm(config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)

        self._init_weights()  # Ensure weights are initialized properly
        self.embedding.weight.requires_grad_(True)  # Force embedding updates

    def _init_weights(self):
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.xavier_uniform_(self.projection.weight)
        if self.bias:
            nn.init.uniform_(self.hidden.bias)
            nn.init.uniform_(self.projection.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.embedding(x)  # Token embeddings (B, T, E)
        # Apply padding mask by zeroing out embeddings for padding tokens
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
        hidden = F.silu(self.hidden(x))  # Non-linear transformation
        hidden = self.dropout(self.norm(hidden))  # Apply LayerNorm + Dropout
        return self.projection(hidden)  # Project back to embedding dim


class MiniTransformer(nn.Module):
    def __init__(self, config: MiniConfig):
        super().__init__()
        self.bias = config.bias

        self.encoder = MiniEncoder(config)
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

    def forward(self, x, mask=None):
        x = self.encoder(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        return self.head(x)
