"""
Copyright © 2023 Austin Berrio
Module: mini.modules.feed_forward
Description: Feedforward blocks for transformer models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini.config import ConfigTransformer


class PositionWiseFeedForward(nn.Module):
    """Standard Feed-Forward Network (FFN) used in transformer layers.

    Uses GELU activation and follows the classic transformer design.
    """

    def __init__(self, config: ConfigTransformer):
        super().__init__()
        self.bias = config.bias
        embed_dim = config.embed_dim  # Input and output size = d_model
        hidden_dim = config.hidden_dim

        # Linear layers for the feed-forward network
        self.fc1 = nn.Linear(embed_dim, hidden_dim, bias=config.bias)
        self.fc2 = nn.Linear(hidden_dim, embed_dim, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feed-forward network."""
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class GatedFeedForward(nn.Module):
    """Gated Feed-Forward Network (GLU-style) with SwiGLU activation.

    Uses SiLU (Swish) activation followed by a gating mechanism.
    Inspired by architectures like GPT-4, PaLM, and Mistral.
    """

    def __init__(self, config: ConfigTransformer):
        super().__init__()
        self.bias = config.bias
        embed_dim = config.embed_dim
        hidden_dim = config.hidden_dim

        # Linear layers for the feed-forward network
        self.w1 = nn.Linear(embed_dim, hidden_dim, bias=config.bias)
        self.w2 = nn.Linear(hidden_dim, embed_dim, bias=config.bias)
        self.w3 = nn.Linear(embed_dim, hidden_dim, bias=config.bias)

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feed-forward network."""
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
