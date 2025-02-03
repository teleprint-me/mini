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

    def as_dict(self) -> dict[str, any]:
        """Returns a dictionary representation of the config."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def from_dict(self, config: dict[str, any]) -> "TransformerConfig":
        """Returns a new TransformerConfig object from a dictionary."""
        return TransformerConfig(**config)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # Gamma

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)


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

        cos_emb, sin_emb = self._precompute_rotary(
            config.head_dim, config.max_seq_len, config.theta
        )
        self.register_buffer("cos_emb", cos_emb)
        self.register_buffer("sin_emb", sin_emb)

    def _precompute_rotary(
        self, dim: int, seq_len: int, theta: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Precompute cosine and sine components for RoPE."""
        dim_half = dim // 2  # RoPE applies to half of the head dimension

        # Compute inverse frequency for rotation
        inv_freqs = 1.0 / (
            theta ** (torch.arange(0, dim_half, dtype=torch.float32) / dim_half)
        )

        # Compute position indices as 1D (seq_len,)
        positions = torch.arange(seq_len, dtype=torch.float32)

        # Compute angles (m * theta), Shape: (seq_len, dim_half)
        angle_rates = torch.einsum("m,d->md", positions, inv_freqs)

        # Compute cos and sin values, Shape: (1, 1, seq_len, dim_half)
        cos_emb = torch.cos(angle_rates).unsqueeze(0).unsqueeze(0)
        sin_emb = torch.sin(angle_rates).unsqueeze(0).unsqueeze(0)

        return cos_emb, sin_emb

    def _apply_rotary(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, T, D = query.shape
        D_half = D // 2  # Ensure we only rotate half of the dimensions

        # Get precomputed RoPE embeddings
        cos_theta = self.cos_emb[:, :, :T, :D_half].expand(B, H, T, D_half)
        sin_theta = self.sin_emb[:, :, :T, :D_half].expand(B, H, T, D_half)

        # Split query and key into even/odd indexed pairs
        q1, q2 = query[..., ::2], query[..., 1::2]
        k1, k2 = key[..., ::2], key[..., 1::2]

        # Apply RoPE transformation
        query_rot = torch.cat(
            [q1 * cos_theta - q2 * sin_theta, q2 * cos_theta + q1 * sin_theta], dim=-1
        )
        key_rot = torch.cat(
            [k1 * cos_theta - k2 * sin_theta, k2 * cos_theta + k1 * sin_theta], dim=-1
        )

        return query_rot, key_rot

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, _ = x.shape
        query, key, value = self.wq(x), self.wk(x), self.wv(x)
        query = query.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        key = (
            key.view(B, T, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
            .repeat(1, self.n_heads // self.n_kv_heads, 1, 1)
        )
        value = (
            value.view(B, T, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
            .repeat(1, self.n_heads // self.n_kv_heads, 1, 1)
        )

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

        self._init_weights()

    def _init_weights(self):
        """Initializes model parameters using best practices for transformers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, RMSNorm):
                module.weight.data.fill_(1.0)

    def forward(self, x, mask=None):
        B, T = x.shape
        x = self.embed(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        return self.head(x)
