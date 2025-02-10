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
        hidden_dim = int(config.ff_dim * config.ff_mult)  # Expand via multiplier

        # Linear layers for the feed-forward network
        self.fc1 = nn.Linear(embed_dim, hidden_dim, bias=config.bias)
        self.fc2 = nn.Linear(hidden_dim, embed_dim, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        # When bias is False, model will not learn an additive bias
        if self.bias:
            torch.nn.init.uniform_(self.fc1.bias, a=-0.1, b=0.1)
            torch.nn.init.uniform_(self.fc2.bias, a=-0.1, b=0.1)

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
        hidden_dim = int(config.ff_dim * config.ff_mult)  # Ensure integer

        # Linear layers for the feed-forward network
        self.w1 = nn.Linear(embed_dim, hidden_dim, bias=config.bias)
        self.w2 = nn.Linear(hidden_dim, embed_dim, bias=config.bias)
        self.w3 = nn.Linear(embed_dim, hidden_dim, bias=config.bias)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform distribution."""
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)
        # When bias is False, model will not learn an additive bias
        if self.bias:
            nn.init.uniform_(self.w1.bias, a=-0.1, b=0.1)
            nn.init.uniform_(self.w2.bias, a=-0.1, b=0.1)
            nn.init.uniform_(self.w3.bias, a=-0.1, b=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feed-forward network."""
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
