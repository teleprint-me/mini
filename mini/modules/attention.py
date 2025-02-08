"""
Copyright © 2023 Austin Berrio
Module: mini.modules.attention
Description: Modular attention mechanisms for neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini.config import ConfigTransformer
from mini.modules.encoding import RotaryEncoding


class BaseAttention(nn.Module):
    def __init__(self, config: ConfigTransformer):
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

    def _attention_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Creates an attention mask of shape [B, 1, T, T] to ignore pad tokens."""
        B, T, _ = x.shape  # Extract batch size and sequence length
        mask = (x[:, :, 0] != self.pad_id).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
        return mask.expand(B, 1, T, T)  # Expand to [B, 1, T, T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the attention mechanism."""
        raise NotImplementedError("Forward method must be implemented by subclasses.")


class SelfAttention(BaseAttention):
    def __init__(self, config: ConfigTransformer):
        super().__init__(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # Batch size, sequence length, embedding dimension

        # Compute Q, K, V matrices
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        # Reshape to multiple heads: [B, T, H, D] -> [B, H, T, D]
        q, k, v = [
            t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            for t in (q, k, v)
        ]

        # Compute scaled dot product attention
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, T, T]

        # Apply softmax and compute output
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)


class CausalAttention(BaseAttention):
    def __init__(self, config: ConfigTransformer):
        super().__init__(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # Batch size, sequence length, embedding dimension

        # Compute Q, K, V matrices
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        # Reshape to multiple heads: [B, T, H, D] -> [B, H, T, D]
        q, k, v = [
            t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            for t in (q, k, v)
        ]

        # Compute scaled dot product attention
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, T, T]

        # Compute and apply mask
        mask = self._attention_mask(x)  # Ensure shape [B, 1, T, T]
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        # Apply softmax and compute output
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)


class RotaryAttention(BaseAttention):
    """Rotary attention mechanism for transformers."""

    def __init__(self, config: ConfigTransformer):
        super().__init__(config)

        # Initialize rotary encoding
        self.rope = RotaryEncoding(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # Batch size, sequence length, embedding dimension

        # Compute Q, K, V matrices
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        # Reshape to multiple heads: [B, T, H, D] -> [B, H, T, D]
        q, k, v = [
            t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            for t in (q, k, v)
        ]

        # Apply RoPE to Q and K
        q, k = self.rope(q, k)

        # Compute scaled dot product attention
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, T, T]

        # Compute and apply mask
        mask = self._attention_mask(x)  # Ensure shape [B, 1, T, T]
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        # Apply softmax and compute output
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)
