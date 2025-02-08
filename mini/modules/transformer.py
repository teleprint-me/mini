"""
Copyright © 2023 Austin Berrio
Module: mini.modules.transformer
Description: Contains the hidden transformer model components.
"""

import torch.nn as nn

from mini.config import ConfigTransformer
from mini.modules.attention import MultiHeadAttention
from mini.modules.feed_forward import FeedForward
from mini.modules.normalization import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: ConfigTransformer):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.norm1 = RMSNorm(config)
        self.norm2 = RMSNorm(config)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x
