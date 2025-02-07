"""
Module: mini.blocks.hidden
Description: Contains the HiddenBlock class for the mini transformer model.
"""

import torch.nn as nn

from mini.blocks.attention import MultiHeadAttention
from mini.blocks.feed_forward import FeedForward
from mini.blocks.norm import RMSNorm
from mini.config import TransformerConfig


class HiddenBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.norm1 = RMSNorm(config)
        self.norm2 = RMSNorm(config)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x
