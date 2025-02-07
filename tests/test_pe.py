"""
Module: tests.pe
Description: This module contains tests for positional encoding.
NOTE: This is tricky because the positional encoding depends on the embedding object. To mitigate this,
we create a fixture to simulate the embedding layer and use it in the positional encoding tests.
"""

import pytest
import torch
import torch.nn as nn


# Need test to reset after each test
@pytest.fixture(scope="session")
def tokens(mini_config, input_tensor) -> torch.nn.Embedding:
    """Fixture to generate embedded token sequences for Positional Encoding (PE) testing."""
    # Simulated token embeddings layer
    embedding = nn.Embedding(
        num_embeddings=mini_config.vocab_size,  # (V) Vocabulary size
        embedding_dim=mini_config.embed_dim,  # (d_e) Embedding dimension
        padding_idx=mini_config.pad_id,  # Automatically ignores pad tokens
    )
    return embedding(input_tensor)  # Output shape is (B, T, C)


@pytest.fixture(scope="module")
def expected_first_row():
    """Fixture to provide expected output data for positional encoding tests."""
    # cos() and sin() oscillations for the first row when input is padding token
    return torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])


@pytest.fixture(scope="module")
def expected_trend():
    """Fixture to provide expected trend data for positional encoding tests."""
    return torch.tensor(
        [
            [0.6926, 0.5657, -1.6573, 0.2519],  # pos=0
            [0.3389, 0.2706, -0.7765, -2.5436],  # pos=1
            [-0.6137, -3.8481, -0.3941, -2.9618],  # pos=2
        ]
    )


def test_pe_shape(positional_encoding, tokens):
    """Ensure positional encoding has the expected shape."""
    # B - batch size, T - sequence length, C - embedding dimension
    pe = positional_encoding(tokens)  # Expected shape is (B, T, C)
    # mini_config has l_max = max_seq_len = 128 and d_e = embed_dim = 256
    assert pe.shape == (1, 128, 256), f"Expected shape (1, 128, 256), got {pe.shape}"


def test_pe_addition(positional_encoding, tokens):
    """Ensure positional encoding correctly modifies input embeddings."""
    pe = positional_encoding(tokens)
    combined = tokens + pe
    assert combined.shape == tokens.shape, "PE addition mismatch"


def test_pe_first_row(positional_encoding, tokens, expected_first_row):
    """Verify that the first row of the PE matrix follows the expected structure."""
    # Positional encodings create a sinusoidal pattern for each position
    pe = positional_encoding._create_positional_encoding()
    assert torch.allclose(
        pe[0, :10], expected_first_row
    ), "First row values do not match expected pattern."


def test_pe_trend(positional_encoding, tokens, expected_trend):
    """Check the trend of the first few rows of the PE matrix."""
    pe = positional_encoding(tokens)

    # If batch-first, remove batch dim
    if pe.dim() == 3:
        pe = pe.squeeze(0)  # Convert (1, T, d_e) → (T, d_e)

    assert torch.allclose(pe[1:4, :4], expected_trend, atol=1e-2), "PE trend mismatch"
