"""
Module: tests.pe
Description: This module contains tests for positional encoding.
"""

import pytest
import torch


def test_pe_shape(positional_encoding):
    """Ensure positional encoding has the expected shape."""
    pe = positional_encoding()
    assert pe.shape == (512, 256), f"Expected shape (512, 256), got {pe.shape}"


def test_pe_first_row(positional_encoding):
    """Verify that the first row of the PE matrix follows the expected structure."""
    pe = positional_encoding()
    expected_first_row = torch.tensor(
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    )
    assert torch.allclose(
        pe[0, :10], expected_first_row
    ), "First row values do not match expected pattern."


def test_pe_trend(positional_encoding):
    """Check the trend of the first few rows of the PE matrix."""
    pe = positional_encoding()
    expected_trend = torch.tensor(
        [
            [0.8415, 0.5403, 0.0998, 0.9950],  # pos=1
            [0.9093, -0.4161, 0.1987, 0.9801],  # pos=2
            [0.1411, -0.9899, 0.2955, 0.9553],  # pos=3
        ]
    )
    assert torch.allclose(pe[1:4, :4], expected_trend, atol=1e-2), "PE trend mismatch"
