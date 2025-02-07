"""
Module: tests.conftest
Description: This module contains configuration and fixtures for testing Mini.
"""

import os

import pytest
import torch
from sentencepiece import SentencePieceProcessor

from mini.transformer.model import (
    MiniConfig,
    MiniEmbedding,
    MiniRuntime,
    PositionalEncoding,
)

# Set PYTHONPATH to current working directory
os.environ["PYTHONPATH"] = os.getcwd()

torch.manual_seed(42)


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
@pytest.fixture(scope="session")
def processor() -> SentencePieceProcessor:
    """Fixture to get the processor to tokenize the text."""
    return SentencePieceProcessor(model_file="models/tokenizer.model")


@pytest.fixture(scope="session")
def mini_runtime(pytestconfig) -> MiniRuntime:
    """Fixture to initialize MiniRuntime with CLI seed/device support."""
    seed = pytestconfig.getoption("--seed")
    runtime = MiniRuntime(seed=seed)
    runtime.seed_all()
    yield runtime  # No explicit cleanup needed


@pytest.fixture(scope="session")
def runtime_device(mini_runtime) -> torch.device:
    """Fixture to get the device to run the model on."""
    return mini_runtime.device_type


@pytest.fixture(scope="session")
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
        pad_id=max(processor.pad_id(), 0),
        dropout=0.1,
        eps=1e-6,
        bias=False,
    )


@pytest.fixture(scope="session")
def positional_encoding(mini_config) -> PositionalEncoding:
    """Fixture for positional encoding model."""
    return PositionalEncoding(mini_config)


@pytest.fixture(scope="session")
def embedding(mini_config) -> MiniEmbedding:
    """Fixture for embedding model."""
    return MiniEmbedding(mini_config)
