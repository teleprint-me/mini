"""
Copyright © 2023 Austin Berrio
Module: mini.models.misty
Author: Austin Berrio
Date: 2025-02-07
Version: 0.1.0
License: AGPL License
URL: https://github.com/teleprint-me/mini
Description:
Misty is a minimal transformer model designed for educational purposes.
It is a simplified version of the transformer architecture, suitable for beginners
to understand the basics of neural network design and training.
"""

import torch.nn as nn

from mini.config import ConfigTransformer
from mini.modules.embedding import PositionalEmbedding
from mini.modules.hidden import HiddenBlock
from mini.modules.normalization import RMSNorm


# Wondering what a good name would be for this class.
# MistyTransformer? or something else? Could use MistyModel or MistyNet.
class MistyTransformer(nn.Module):
    def __init__(self, config: ConfigTransformer):
        super().__init__()
        self.bias = config.bias

        self.embedding = PositionalEmbedding(config)
        self.blocks = nn.ModuleList(
            [HiddenBlock(config) for _ in range(config.num_layers)]
        )
        self.norm = RMSNorm(config)
        self.head = nn.Linear(config.embed_dim, config.vocab_size)

        self.max_seq_len = config.max_seq_len
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.head.weight)
        if self.bias is not None:
            nn.init.uniform_(self.head.bias)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)
