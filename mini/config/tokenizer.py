"""
Copyright © 2023 Austin Berrio
Module: mini.config.tokenizer
Description: Configuration settings for training a SentencePiece tokenizer.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from mini.config.base import ConfigBase


@dataclass
class ConfigTokenizer(ConfigBase):
    """Configuration for SentencePiece tokenizer training."""

    # Paths
    model_path: Path = Path("models/tokenizer.model")
    input: Path = Path("data/mini-owl-fairy.md")

    # Basic training parameters
    model_prefix: str = "tokenizer"
    model_type: str = "bpe"  # Available: 'unigram', 'bpe', 'char', 'word'
    vocab_size: int = 1000
    character_coverage: float = 1.0

    # Data processing parameters
    input_sentence_size: int | None = None  # Limit training sentences
    max_sentence_length: int = 4192
    shuffle_input_sentence: bool = False
    split_by_whitespace: bool = False

    # Special tokens
    user_defined_symbols: list[str] = field(default_factory=list)
    control_symbols: list[str] = field(default_factory=list)

    # Padding configuration
    enable_padding: bool = False
    pad_id: int = 3  # Applied only if `enable_padding=True`

    # Subword and tokenization options
    byte_fallback: bool = False
    allow_whitespace_only_pieces: bool = False
    remove_extra_whitespaces: bool = False
    split_digits: bool = False

    def as_dict(self) -> Dict[str, Any]:
        """Returns a dictionary of valid training parameters for SentencePieceTrainer."""
        config = super().as_dict()

        # Ensure input paths are converted to strings
        config["input"] = str(self.input)
        config["model_path"] = str(self.model_path)

        # Remove `enable_padding`, handle `pad_id` dynamically
        config.pop("enable_padding", None)
        if not self.enable_padding:
            config["pad_id"] = -1  # Disable padding

        return config

    def list_model_types(self) -> tuple[str, ...]:
        """Return available tokenizer model types."""
        return ("unigram", "bpe", "char", "word")

    def timestamp_model_prefix(self) -> str:
        """Return a timestamped model prefix."""
        return f"{self.model_prefix}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
