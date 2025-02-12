"""
Copyright © 2023 Austin Berrio
Module: mini.modules.mlp
Description: Multi-layer perceptron (MLP) module for mini projects.
"""

import torch
import torch.nn as nn

from mini.config import ConfigTransformer


# Multilayer feedforward networks are universal approximators:
# https://dl.acm.org/doi/abs/10.5555/70405.70408
class MultiLayerPerceptron(nn.Module):
    """A simple multi-layer perceptron for encoded transformations."""

    def __init__(self, config: ConfigTransformer):
        super().__init__()
        self.config = config  # Ensure config is assigned before calling _init_layers
        self._init_layers()  # Initialize layers
        self._init_weights()  # Initialize weights

    def _init_layers(self):
        """Defines MLP layers for encoded transformations."""
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
        # Final projection to match token encoded space
        self.projection = nn.Linear(self.config.hidden_dim, self.config.embed_dim)
        # Dropout & LayerNorm for regularization
        self.norm = nn.LayerNorm(self.config.embed_dim)
        self.dropout = nn.Dropout(self.config.dropout)

    def _init_weights(self):
        """Initializes all weights using Xavier uniform distribution."""
        for layer in self.layers:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.uniform_(module.bias, a=-0.1, b=0.1)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.uniform_(module.bias, a=-0.1, b=0.1)

        # Initialize projection layer
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.uniform_(self.projection.bias, a=-0.1, b=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the perceptron model layers."""
        for layer in self.layers:
            x = layer(x)
        # Project output with shape (B, T, C)
        x = self.projection(x)
        # Residual connection and layer normalization
        return self.dropout(self.norm(x)) + x
