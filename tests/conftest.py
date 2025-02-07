"""
Module: tests.conftest
Description: This module contains configuration and fixtures for testing.
"""

import os

import pytest
import torch
from sentencepiece import SentencePieceProcessor

from mini.transformer.model import (
    MiniConfig,
    MiniEmbedding,
    MiniRuntime,
    PositionalEmbedding,
)

# Set PYTHONPATH to current working directory
os.environ["PYTHONPATH"] = os.getcwd()

# --- Pytest Configuration ---


def pytest_addoption(parser):
    """Add custom CLI options for pytest."""
    parser.addoption(
        "--private",
        action="store_true",
        default=False,
        help="Run tests marked as private.",
    )
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="Run slow tests.",
    )
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        help="Device to run tests on (Default: cpu).",
    )
    parser.addoption(
        "--seed",
        action="store",
        type=int,
        default=42,
        help="Random seed for reproducibility (Default: 42).",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "private: Tests that require --private option to run."
    )
    config.addinivalue_line("markers", "slow: Tests that are slow to run.")


def pytest_collection_modifyitems(config, items):
    """Modify collected test items based on CLI options."""
    if not config.getoption("--private"):
        private_marker = pytest.mark.skip(reason="Use --private to run this test.")
        for item in items:
            if "private" in item.keywords:
                item.add_marker(private_marker)

    if not config.getoption("--slow"):
        slow_marker = pytest.mark.skip(reason="Use --slow to run this test.")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(slow_marker)


# --- Fixtures ---


# Need to seed torch with mini_runtime.seed_all()
@pytest.fixture
def mini_runtime() -> MiniRuntime:
    """Fixture to initialize MiniRuntime."""
    runtime = MiniRuntime(seed=42)
    runtime.seed_all()
    return runtime


@pytest.fixture
def runtime_device(mini_runtime) -> torch.device:
    """Fixture to get the device to run the model on."""
    return mini_runtime.device_type


# Create a fixture for the processor
@pytest.fixture
def processor() -> SentencePieceProcessor:
    """Fixture to get the processor to tokenize the text."""
    return SentencePieceProcessor(model_file="models/tokenizer.model")


# Create a fixture for the configuration object
@pytest.fixture
def mini_config(processor) -> MiniConfig:
    """Fixture to get the configuration object."""
    return MiniConfig(
        vocab_size=processor.vocab_size(),
        embed_dim=256,
        num_heads=8,
        head_dim=32,
        num_layers=4,
        ff_dim=512,
        max_seq_len=128,
        pad_id=max(processor.get_piece_ids(), 0),
        dropout=0.1,
        eps=1e-6,
        bias=False,
    )


# Create a fixture for the positional encoding model
@pytest.fixture
def positional_encoding(config):
    return PositionalEmbedding(config)


# Create a fixture for the embedding model
@pytest.fixture
def embedding(config):
    return MiniEmbedding(config)
