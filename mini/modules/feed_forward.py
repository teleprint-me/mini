"""
Copyright © 2023 Austin Berrio
Module: mini.modules.feed_forward
Description: Feedforward blocks for transformer models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini.config import ConfigTransformer


class FeedForward(nn.Module):
    def __init__(self, config: ConfigTransformer):
        super().__init__()
        self.bias = config.bias

        self.w1 = nn.Linear(config.embed_dim, config.ff_dim, bias=config.bias)
        self.w2 = nn.Linear(config.ff_dim, config.embed_dim, bias=config.bias)
        self.w3 = nn.Linear(config.embed_dim, config.ff_dim, bias=config.bias)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)
        if self.bias:
            nn.init.uniform_(self.w1.bias)
            nn.init.uniform_(self.w2.bias)
            nn.init.uniform_(self.w3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
