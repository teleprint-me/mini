"""
Module: tests.pe
Description: This module contains tests for positional encoding.
NOTE: This is tricky because the positional encoding depends on the embedding object. To mitigate this,
we create a fixture to simulate the embedding layer and use it in the positional encoding tests.
"""

import pytest
import torch
import torch.nn as nn


@pytest.fixture
def embedding(mini_config) -> torch.nn.Embedding:
    """Fixture to create an embedding layer."""
    return nn.Embedding(
        num_embeddings=mini_config.vocab_size,  # (V) Vocabulary size
        embedding_dim=mini_config.embed_dim,  # (d_e) Embedding dimension
        padding_idx=mini_config.pad_id,  # Automatically ignores pad tokens
    )


@pytest.fixture
def input_data(processor, mini_config, embedding) -> torch.Tensor:
    """Fixture to generate embedded token sequences for PE testing."""
    # Encode text to token IDs (T = v \in [0, V-1] for V tokens)
    input_ids = processor.encode("The quick brown fox jumped over the")
    # Pad sequence (l_max = l \in [0, l_max] for l tokens)
    add_padding = mini_config.max_seq_len - len(input_ids)
    padded_ids = input_ids + ([mini_config.pad_id] * add_padding)
    # Convert to tensor (B, T) where B = 1 batch and T = l_max tokens
    tensor_ids = torch.tensor([padded_ids], dtype=torch.long)
    # Simulated token embeddings (B, T, d_e)
    return embedding(tensor_ids)


@pytest.fixture
def expected_first_row():
    """Fixture to provide expected output data for positional encoding tests."""
    # cos() and sin() oscillations for the first row when input is padding token
    return torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])


@pytest.fixture
def expected_trend():
    """Fixture to provide expected trend data for positional encoding tests."""
    # NOTE: This needs to be fixed to match the actual trend of the positional encoding.
    return torch.tensor(
        [
            [0.8415, 0.5403, 0.0998, 0.9950],  # pos=1
            [0.9093, -0.4161, 0.1987, 0.9801],  # pos=2
            [0.1411, -0.9899, 0.2955, 0.9553],  # pos=3
        ]
    )


def test_pe_shape(positional_encoding, input_data):
    """Ensure positional encoding has the expected shape."""
    # B - batch size, T - sequence length, C - embedding dimension
    pe = positional_encoding(input_data)  # Expected shape is (B, T, C)
    # mini_config has l_max = max_seq_len = 128 and d_e = embed_dim = 256
    assert pe.shape == (1, 128, 256), f"Expected shape (1, 128, 256), got {pe.shape}"


def test_pe_addition(positional_encoding, input_data):
    """Ensure positional encoding correctly modifies input embeddings."""
    pe = positional_encoding(input_data)
    combined = input_data + pe
    assert combined.shape == input_data.shape, "PE addition mismatch"


def test_pe_first_row(positional_encoding, input_data, expected_first_row):
    """Verify that the first row of the PE matrix follows the expected structure."""
    pe = positional_encoding._create_positional_encoding()
    assert torch.allclose(
        pe[0, :10], expected_first_row
    ), "First row values do not match expected pattern."


def test_pe_trend(positional_encoding, input_data, expected_trend):
    """Check the trend of the first few rows of the PE matrix."""
    # Is seeding working in conftest? Values keep changing.
    pe = positional_encoding(input_data)
    print(f"PE shape: {pe.shape}")
    print(f"PE first few values: {pe[1:4, :4]}")
    print(f"Expected trend: {expected_trend}")

    # If batch-first, remove batch dim
    if pe.dim() == 3:
        pe = pe.squeeze(0)  # Remove batch dimension if present

    assert torch.allclose(pe[1:4, :4], expected_trend, atol=1e-2), "PE trend mismatch"
