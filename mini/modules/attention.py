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
        self.dtype = config.dtype

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
        size = (self.max_seq_len, self.max_seq_len)
        return torch.ones(size, device=self.device, dtype=self.dtype)

    def _causal_mask(self) -> torch.Tensor:
        """Create a causal mask to prevent attending to future tokens."""
        dtype = self.dtype  # Get dtype from config
        min_value = torch.finfo(dtype).min  # Get smallest representable value
        size = (self.config.max_seq_len, self.config.max_seq_len)  # [T, T] matrix
        mask = torch.full(size, min_value, device=self.device, dtype=dtype)
        return torch.triu(mask, diagonal=1)  # Keep upper triangle for causal mask


class BaseAttention(nn.Module):
    def __init__(self, config: ConfigTransformer):
        super().__init__()
        self.pad_id = config.pad_id
        self.bias = config.bias
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = config.scale
        self.dtype = config.dtype

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

    def _split_heads(
        self, d_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Q, K, V projections."""
        return self.wq(d_in), self.wk(d_in), self.wv(d_in)

    def _reshape_heads(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, B: int, T: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reshape heads for multi-head attention."""
        return [
            t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            for t in (q, k, v)
        ]

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Computes scaled dot-product attention with optional masking."""
        # Compute scaled dot-product attention
        d_attn = (q @ k.transpose(-2, -1)) * self.scale
        # Apply mask if provided
        if mask is not None:
            d_attn = d_attn.masked_fill(mask == 0, torch.finfo(self.dtype).min)
        # Apply softmax to get attention weights
        d_attn = F.softmax(d_attn, dim=-1)
        # Apply multi-head attention weights to values
        return d_attn @ v  # [B, num_heads, T, head_dim]

    def _merge_heads(
        self, d_attn: torch.Tensor, B: int, T: int, C: int
    ) -> torch.Tensor:
        """Reshape concatenated multi-head output back to [B, T, C]."""
        return d_attn.transpose(1, 2).contiguous().view(B, T, C)

    def forward(self, d_in: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of the attention mechanism. Input shape is [B, T, C]."""
        raise NotImplementedError("Forward method must be implemented by subclasses.")


class SelfAttention(BaseAttention):
    def __init__(self, config: ConfigTransformer):
        super().__init__(config)

    def forward(self, d_in: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Get batch size, sequence length, and embedding dimension
        B, T, C = d_in.shape
        # Compute Q, K, V projections
        q, k, v = self._split_heads(d_in)
        # Reshape to multiple heads: [B, T, d_model] → [B, num_heads, T, head_dim]
        q, k, v = self._reshape_heads(q, k, v, B, T)
        # Compute scaled dot-product attention
        d_attn = self._scaled_dot_product_attention(q, k, v, mask)
        # Reshape: concatenate heads → [B, T, d_model]
        d_out = self._merge_heads(d_attn, B, T, C)
        # Final linear projection
        return self.wo(d_out)


class RotaryAttention(BaseAttention):
    """Rotary attention mechanism for transformers."""

    def __init__(self, config: ConfigTransformer):
        super().__init__(config)
        # Initialize rotary encoding
        self.rope = RotaryEncoding(config)

    def forward(self, d_in: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Get batch size, sequence length, and embedding dimension
        B, T, C = d_in.shape
        # Compute Q, K, V matrices
        q, k, v = self._split_heads(d_in)
        # Apply RoPE encoding to Q and K before reshaping heads
        q, k = self.rope(q, k)
        # Reshape to multiple heads: [B, T, d_model] → [B, num_heads, T, head_dim]
        q, k, v = self._reshape_heads(q, k, v, B, T)
        # Compute scaled dot-product attention
        d_attn = self._scaled_dot_product_attention(q, k, v, mask)
        # Reshape: concatenate heads → [B, T, d_model]
        d_out = self._merge_heads(d_attn, B, T, C)
        # Final linear projection
        return self.wo(d_out)
