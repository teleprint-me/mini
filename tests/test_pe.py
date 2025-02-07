"""
Module: tests.pe
Description: This module contains tests for positional encoding.
"""

import pytest
import torch


@pytest.fixture
def input_data():
    """Fixture to provide input data for positional encoding tests."""
    # max seq len is 128 $l_{max}$, embedding dim is 256 $\mathbb{R}^{d_e}$
    return torch.randn(128, 256)  # Shape is (seq_len, embed_dim)


@pytest.fixture
def expected_first_row():
    """Fixture to provide expected output data for positional encoding tests."""
    # COS and SIN oscillations for the first row when input is padding token
    return torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])


@pytest.fixture
def expected_trend():
    """Fixture to provide expected trend data for positional encoding tests."""
    # This should be the expected trend data for the given input data
    return torch.tensor(
        [
            [0.8415, 0.5403, 0.0998, 0.9950],  # pos=1
            [0.9093, -0.4161, 0.1987, 0.9801],  # pos=2
            [0.1411, -0.9899, 0.2955, 0.9553],  # pos=3
        ]
    )


def test_pe_shape(positional_encoding, input_data):
    """Ensure positional encoding has the expected shape."""
    pe = positional_encoding(input_data)  # (batch_size, seq_len, d_model)
    assert pe.shape == (1, 128, 256), f"Expected shape (1, 128, 256), got {pe.shape}"


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
