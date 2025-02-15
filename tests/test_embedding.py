"""
Module: tests.embedding
Description: This module contains unit tests for the embedding functionality.
"""

import pytest
import torch
from sentencepiece import SentencePieceProcessor

from mini.transformer.model import MiniConfig, MiniEmbedding


# Test cases for MiniEmbedding class
def test_mini_embedding_init(embedding: MiniEmbedding):
    """Test the initialization of MiniEmbedding class."""
    assert embedding.embedding_dim == 256
    assert embedding.vocab_size == 32000
    assert embedding.weight.shape == (32000, 256)


def test_mini_embedding_forward():
    """Test the forward pass of MiniEmbedding class."""
    embedding = MiniEmbedding(vocab_size=1000, embedding_dim=512)
    input_ids = torch.randint(0, 1000, (32, 10))
    output = embedding(input_ids)
    assert output.shape == (32, 10, 512)


# Test cases for MiniEmbedding class with padding_idx
def test_mini_embedding_init_with_padding_idx():
    """Test the initialization of MiniEmbedding class with padding_idx."""
    embedding = MiniEmbedding(vocab_size=1000, embedding_dim=512, padding_idx=0)
    assert embedding.padding_idx == 0
    assert embedding.embedding.weight.shape == (1000, 512)
    assert (
        embedding.embedding.weight[0].sum() == 0
    )  # Check if the padding_idx embedding is zero
