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


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # Gamma parameter

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * self._norm(x.float()).type_as(x)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.bias = bias
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform__(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)
        if self.bias:
            nn.init.uniform_(self.w1.bias)
            nn.init.uniform_(self.w2.bias)
            nn.init.uniform_(self.w3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.scale = self.head_dim**-0.5

        self.wq = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        self.wk = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        self.wv = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        self.wo = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)

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


class MiniEmbedding(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocab_size, config.embed_dim, padding_idx=config.pad_id
        )
        self.hidden = nn.Linear(config.embed_dim, config.max_seq_len)
        self.projection = nn.Linear(config.max_seq_len, config.embed_dim)

        self.norm = nn.LayerNorm(config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def _init_weights(self):
        nn.init.xavier_uniform__(self.hidden.weight)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.uniform_(self.hidden.bias)
        nn.init.uniform_(self.projection.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        hidden = self.dropout(self.activation(self.hidden(x)))
        hidden = self.norm(hidden)
        out = self.projection(hidden).mean(dim=1)
        return F.normalize(out, p=2, dim=1)


class MiniTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embedding = MiniEmbedding(config)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size)
        self.max_seq_len = config.max_seq_len
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.uniform_(module.weight, -0.1, 0.1)
            elif isinstance(module, RMSNorm):
                module.weight.data.fill_(1.0)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        return self.head(x)
