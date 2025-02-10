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


class AttentionMask:
    """Generates attention masks for input sequences."""

    def __init__(self, config: ConfigTransformer):
        self.config = config
        self.pad_id = config.pad_id
        self.max_seq_len = config.max_seq_len
        self.device = config.device

    def __call__(
        self, input_ids: torch.Tensor, mask_type: str = "causal"
    ) -> torch.Tensor:
        """Combine padding and attention masks."""
        pad_mask = self._get_pad_mask(input_ids)
        attn_mask = self._get_attention_mask(mask_type)
        return pad_mask + attn_mask  # Add masks together

    def _get_pad_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create a padding mask of shape [B, 1, 1, T] from input IDs before they are embedded."""
        return (input_ids == self.pad_id).unsqueeze(1).unsqueeze(2)

    def _get_attention_mask(self, mask_type: str) -> torch.Tensor:
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

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Computes scaled dot-product attention with optional masking."""
        # Compute scaled dot-product attention
        d_attn = (q @ k.transpose(-2, -1)) * self.scale
        # Apply mask if provided
        if mask is not None:
            d_attn = d_attn.masked_fill(mask == float("-inf"), float("-inf"))
        # Apply softmax to get attention weights
        d_attn = F.softmax(d_attn, dim=-1)
        # Apply multi-head attention weights to values
        return d_attn @ v  # [B, num_heads, T, head_dim]

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of the attention mechanism. Input shape is [B, T, C]."""
        raise NotImplementedError("Forward method must be implemented by subclasses.")


class SelfAttention(BaseAttention):
    def __init__(self, config: ConfigTransformer):
        super().__init__(config)

    def forward(self, d_in: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Get batch size, sequence length, and embedding dimension
        B, T, C = d_in.shape
        # Compute Q, K, V projections
        q, k, v = self.wq(d_in), self.wk(d_in), self.wv(d_in)
        # Reshape to multiple heads: [B, T, d_model] → [B, num_heads, T, head_dim]
        q, k, v = [
            t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            for t in (q, k, v)
        ]
        # Compute scaled dot-product attention
        d_attn = self._scaled_dot_product_attention(q, k, v, mask)
        # Reshape: concatenate heads → [B, T, d_model]
        d_out = d_attn.transpose(1, 2).contiguous().view(B, T, C)
        # Final linear projection
        return self.wo(d_out)


class RotaryAttention(BaseAttention):
    """Rotary attention mechanism for transformers."""

    def __init__(self, config: ConfigTransformer):
        super().__init__(config)
        # Initialize rotary encoding
        self.rope = RotaryEncoding(config)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Get batch size, sequence length, and embedding dimension
        B, T, C = x.shape
        # Compute Q, K, V matrices
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        # Apply RoPE encoding to Q and K before reshaping heads
        q, k = self.rope(q, k)
        # Reshape to multiple heads: [B, T, H, D] -> [B, H, T, D]
        q, k, v = [
            t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            for t in (q, k, v)
        ]
        # Compute scaled dot-product attention
        d_attn = self._scaled_dot_product_attention(q, k, v, mask)
        # Reshape: concatenate heads → [B, T, d_model]
        d_out = d_attn.transpose(1, 2).contiguous().view(B, T, C)
        # Final linear projection
        return self.wo(d_out)
