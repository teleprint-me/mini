"""
Copyright © 2023 Austin Berrio
Module: mini.modules.rmsnorm
Description: RMSNorm layer for transformer models.
"""

import torch
import torch.nn as nn

from mini.config import ConfigTransformer


class RMSNorm(nn.Module):
    def __init__(self, config: ConfigTransformer):
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.ones(config.embed_dim))  # Gamma parameter

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * self._norm(x.float()).type_as(x)
