"""
Copyright © 2023 Austin Berrio

Module: mini.models.valerie
Author: Austin Berrio
Date: 2025-02-11
Version: 0.1.0
License: AGPL License
URL: https://github.com/teleprint-me/mini

Description:
Valerie is a minimal gated transformer model with rotary attention and
feed-forward networks (FFN) layers designed for educational purposes.
It is a simplified version of the BERT, GPT, Llama, and Mistral models,
suitable for beginners to understand the basics of rotary positional encodings,
learnable embeddings, and gated feed forward networks.
"""

import torch.nn as nn

from mini.config.transformer import ConfigTransformer
from mini.modules.attention import AttentionMask
from mini.modules.embedding import BertEmbedding
from mini.modules.normalization import RMSNorm
from mini.modules.transformer import GatedBlock


# RoFormer: Enhanced Transformer with Rotary Position Embeddings https://arxiv.org/abs/2104.09864
class ValerieModel(nn.Module):
    """Minimal gated transformer model with rotary attention and FFN layers."""

    def __init__(self, config: ConfigTransformer):
        super().__init__()
        self.config = config
        self.bias = config.bias
        self.max_seq_len = config.max_seq_len

        self.mask = AttentionMask(config)
        self.embedding = BertEmbedding(config)
        self.transformer = nn.ModuleList(
            [GatedBlock(config) for _ in range(config.num_layers)]
        )
        self.norm = RMSNorm(config)
        self.head = nn.Linear(config.embed_dim, config.vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.head.weight)
        if self.bias:
            nn.init.uniform_(self.head.bias, a=-0.1, b=0.1)

    def forward(self, input_ids):
        # Attention mask
        mask = self.mask(input_ids)
        # Embedding layer
        x = self.embedding(input_ids)
        # Rotary attention and FFN layers
        for block in self.transformer:
            x = block(x, mask)
        # Final normalization and linear layer
        x = self.norm(x)
        return self.head(x)
