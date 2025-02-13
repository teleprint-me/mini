"""
Module: mini.config.generator
Description: Configuration classes for text generation.
NOTE: ConfigDevice inherits from ConfigBase.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import regex as re
from sentencepiece import SentencePieceProcessor

from mini.config.device import ConfigDevice

if TYPE_CHECKING:
    from mini.engine import EngineSampler, EngineState
else:
    # Forward declarations to mitigate circular dependency.
    # Actual implementations are defined in mini.engine.state and mini.engine.sampler
    class EngineState:
        """Placeholder for EngineState. Defined in mini.engine.state."""

        pass

    class EngineSampler:
        """Placeholder for EngineSampler. Defined in mini.engine.sampler."""

        pass


# Default GPT-2 style pre-tokenizer regex
DEFAULT_PRETOKENIZER = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    re.IGNORECASE,
)


@dataclass
class ConfigSampler(ConfigDevice):
    """Configuration for sampling strategies in text generation."""

    pad_id: int = -1
    top_k: int = 50
    top_p: float = 0.9
    temperature: float = 0.8
    repetition_penalty: float = 1.2
    greedy: bool = False  # Greedy decoding mode
    verbose: bool = False  # Enable debug mode

    def __post_init__(self):
        # Ensure pad_id is non-negative.
        self.pad_id = max(self.pad_id, 0)
        # Initialize the device.
        self.set_device()


@dataclass
class ConfigGenerator(ConfigDevice):
    """Configuration for text generation."""

    state: Optional["EngineState"] = None
    sampler: Optional["EngineSampler"] = None
    processor: Optional[SentencePieceProcessor] = None
    pre_tokenizer: re.Pattern = DEFAULT_PRETOKENIZER  # Directly set default
