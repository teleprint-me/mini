"""
Copyright © 2023 Austin Berrio
Module: mini.model.transformer
Description: A simple transformer model for quick and easy training.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        n_kv_heads: int,
        max_seq_len: int,
        theta: float = 10000.0,
        bias: bool = False,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.repeats = self.n_heads // self.n_kv_heads
        self.scale = self.head_dim**-0.5

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=bias)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=bias)

        # Precompute RoPE frequencies
        self.register_buffer(
            "rope", self._precompute_rotary(head_dim, max_seq_len, theta)
        )

    def _precompute_rotary(self, dim: int, seq_len: int, theta: float) -> torch.Tensor:
        inv_freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, inv_freqs)
        return torch.polar(torch.ones_like(freqs), freqs)

    def _apply_rotary(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        rope: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_complex = torch.view_as_complex(
            query.float().reshape(*query.shape[:-1], -1, 2)
        )
        key_complex = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))
        rope = rope[:, None, :]
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

        # Apply RoPE
        query, key = self._apply_rotary(query, key, self.rope[:T])

        attn_weights = (query @ key.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.masked_fill(attn_weights.isnan(), 0)

        attn_output = (attn_weights @ value).transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(attn_output)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        head_dim,
        n_kv_heads,
        ff_dim,
        max_seq_len,
        theta=10000.0,
        bias=False,
    ):
        super().__init__()
        self.attn = Attention(
            dim, n_heads, head_dim, n_kv_heads, max_seq_len, theta, bias
        )
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.ff = FeedForward(dim, ff_dim)

    def forward(self, x, mask=None):
        rope = self.attn.rope[: x.shape[1]]  # Slice to sequence length
        x = self.norm1(x + self.attn(x, rope, mask))
        x = self.norm2(x + self.ff(x))
        return x


class MiniTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        head_dim,
        num_layers,
        ff_dim,
        max_seq_len,
        pad_id,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim, num_heads, head_dim, num_heads, ff_dim, max_seq_len
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, x, mask=None):
        B, T = x.shape
        x = self.embed(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        return self.head(x)
