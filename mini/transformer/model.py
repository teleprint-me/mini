"""
Copyright © 2023 Austin Berrio
Module: mini.transformer.model
Description: A simple transformer model for quick and easy training.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    vocab_size: int
    embed_dim: int
    num_heads: int
    head_dim: int
    num_layers: int
    ff_dim: int
    max_seq_len: int
    pad_id: int
    theta: float = 10000.0
    bias: bool = False


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_heads = config.num_heads
        self.head_dim = config.head_dim
        self.n_kv_heads = self.n_heads // 2
        self.repeats = self.n_heads // self.n_kv_heads
        self.scale = self.head_dim**-0.5

        self.wq = nn.Linear(
            config.embed_dim, self.n_heads * self.head_dim, bias=config.bias
        )
        self.wk = nn.Linear(
            config.embed_dim, self.n_kv_heads * self.head_dim, bias=config.bias
        )
        self.wv = nn.Linear(
            config.embed_dim, self.n_kv_heads * self.head_dim, bias=config.bias
        )
        self.wo = nn.Linear(
            self.n_heads * self.head_dim, config.embed_dim, bias=config.bias
        )

        self.register_buffer(
            "rope",
            self._precompute_rotary(config.head_dim, config.max_seq_len, config.theta),
        )

    def _precompute_rotary(self, dim: int, seq_len: int, theta: float) -> torch.Tensor:
        inv_freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, inv_freqs)
        return torch.polar(torch.ones_like(freqs), freqs)

    def _apply_rotary(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_complex = torch.view_as_complex(
            query.float().reshape(*query.shape[:-1], -1, 2)
        )
        key_complex = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))
        rope = self.rope[: query.shape[1]]
        query_real = torch.view_as_real(query_complex * rope).flatten(-2)
        key_real = torch.view_as_real(key_complex * rope).flatten(-2)
        return query_real.type_as(query), key_real.type_as(key)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, _ = x.shape
        query, key, value = self.wq(x), self.wk(x), self.wv(x)
        query = query.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        query, key = self._apply_rotary(query, key)
        attn_weights = (query @ key.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.masked_fill(attn_weights.isnan(), 0)

        attn_output = (attn_weights @ value).transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn = Attention(config)
        self.norm1 = RMSNorm(config.embed_dim)
        self.norm2 = RMSNorm(config.embed_dim)
        self.ff = FeedForward(config.embed_dim, config.ff_dim)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.attn(x, mask))
        x = self.norm2(x + self.ff(x))
        return x


class MiniTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embed = nn.Embedding(
            config.vocab_size, config.embed_dim, padding_idx=config.pad_id
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size)
        self.max_seq_len = config.max_seq_len

    def forward(self, x, mask=None):
        B, T = x.shape
        x = self.embed(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        return self.head(x)
