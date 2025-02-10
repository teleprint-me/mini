"""
Copyright © 2023 Austin Berrio
Module: mini.modules.transformer
Description: Contains the hidden transformer model components.
"""

import torch.nn as nn
from torch.nn import LayerNorm

from mini.config import ConfigTransformer
from mini.modules.attention import RotaryAttention, SelfAttention
from mini.modules.feed_forward import GatedFeedForward, PositionWiseFeedForward
from mini.modules.normalization import RMSNorm


class PositionWiseBlock(nn.Module):
    def __init__(self, config: ConfigTransformer):
        super().__init__()
        self.attn = SelfAttention(config)
        self.norm1 = LayerNorm()
        self.norm2 = LayerNorm()
        self.ff = PositionWiseFeedForward(config)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x


class GatedBlock(nn.Module):
    def __init__(self, config: ConfigTransformer):
        super().__init__()
        self.attn = RotaryAttention(config)
        self.norm1 = RMSNorm(config)
        self.norm2 = RMSNorm(config)
        self.ff = GatedFeedForward(config)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x
