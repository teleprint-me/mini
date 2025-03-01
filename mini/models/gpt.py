"""
Full GPT Language Model in a single file.

Reference: https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from mini.config.transformer import ConfigTransformer


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, config: ConfigTransformer):
        super().__init__()

        self.num_heads = config.num_heads
        self.embed_dim = config.embed_dim

        # Key, query, value projections for all heads in a single layer
        self.c_attn = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        self.c_proj = nn.Linear(config.embed_dim, config.embed_dim)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask to prevent attention to future tokens
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).view(
                1, 1, config.max_seq_len, config.max_seq_len
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding size

        # Compute Q, K, V projections
        q, k, v = self.c_attn(x).split(self.embed_dim, dim=2)

        # Reshape for multi-head attention
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Apply attention weights
        y = att @ v

        # Reshape back
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


# NOTE: I think this is what modern architectures label as the FeedForward.
# Need to cross check between Llama and Mistral implementations.
class MLP(nn.Module):
    """Multi-Layer Perceptron (MLP) block for transformers."""

    def __init__(self, config: ConfigTransformer):
        super().__init__()

        hidden_dim = int(config.ff_mult * config.embed_dim)

        self.c_fc = nn.Linear(config.embed_dim, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, config.embed_dim)
        self.act = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """A standard Transformer block"""

    def __init__(self, config: ConfigTransformer):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.embed_dim)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.embed_dim)
        self.mlp = MLP(config)  # Use the standalone MLP

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT Language Model"""

    def __init__(self, config: ConfigTransformer):
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.wte = nn.Embedding(config.vocab_size, config.embed_dim)
        self.wpe = nn.Embedding(config.max_seq_len, config.embed_dim)
        self.drop = nn.Dropout(config.dropout)
        self.transformer = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx):
        """Forward pass for Transformer"""
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.max_seq_len
        ), f"Cannot forward sequence of length {t}, max length is {self.max_seq_len}"

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # (1, t)

        # Token & Positional Embeddings
        tok_emb = self.wte(idx)  # (b, t, embed_dim)
        pos_emb = self.wpe(pos)  # (1, t, embed_dim)
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.transformer:
            x = block(x)

        x = self.ln_f(x)  # Final LayerNorm
        return self.lm_head(x)
