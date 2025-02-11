"""
Copyright © 2023 Austin Berrio
Module: mini.modules.mlp
Description: Multi-layer perceptron (MLP) module for mini projects.
"""

import torch
import torch.nn as nn

from mini.config import ConfigTransformer


class MLPEmbedding(nn.Module):
    """A simple multi-layer perceptron for embedding transformation."""

    def __init__(self, config: ConfigTransformer):
        super().__init__()
        self.config = config  # Ensure config is assigned before calling _init_layers
        self._init_layers()  # Initialize layers
        # Final projection to match token embedding space
        self.projection = nn.Linear(config.max_seq_len, config.embed_dim)
        # Dropout & LayerNorm for regularization
        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        # Initialize weights
        self._init_weights()

    def _init_layers(self):
        """Defines MLP layers for embedding transformation."""
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        self.config.embed_dim if i == 0 else self.config.hidden_dim,
                        self.config.hidden_dim,
                    ),
                    nn.SiLU(),
                    nn.LayerNorm(self.config.hidden_dim),
                    nn.Dropout(self.config.dropout),
                )
                for i in range(self.config.num_mlp_layers)
            ]
        )

    def _init_weights(self):
        """Initializes all weights using Xavier uniform distribution."""
        for layer in self.layers:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.uniform_(module.bias, a=-0.1, b=0.1)
                elif isinstance(module, nn.LayerNorm):
                    if module.bias is not None:
                        nn.init.uniform_(module.bias, a=-0.1, b=0.1)

        # Initialize projection layer
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.uniform_(self.projection.bias, a=-0.1, b=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the embedding layer."""
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)  # Normalize before dropout
        x = self.dropout(x)
        return self.projection(x)  # (B, T, C) output
