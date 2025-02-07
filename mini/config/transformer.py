"""
Module: mini.config.transformer
Description: Configuration settings for the transformer model.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional

import regex as re
from sentencepiece import SentencePieceProcessor

from mini.config.base import BaseConfig
from mini.config.runtime import RuntimeConfig

if TYPE_CHECKING:
    from mini.engine import MiniSampler, MiniState
else:
    # Forward declarations to mitigate circular dependency.
    # Actual implementations are defined in mini.engine.state and mini.engine.sampler
    class MiniState:
        """Placeholder for MiniState. Defined in mini.engine.state."""

        pass

    class MiniSampler:
        """Placeholder for MiniSampler. Defined in mini.engine.sampler."""

        pass


# Default GPT-2 style pre-tokenizer regex
DEFAULT_PRETOKENIZER = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    re.IGNORECASE,
)


@dataclass
class TransformerConfig(BaseConfig):
    """Configuration for transformer model architecture."""

    # Encoder and Embedding
    pad_id: int = -1  # Handle padding in embeddings
    vocab_size: int = 32000  # Number of unique tokens in the vocabulary
    max_seq_len: int = 128  # Max positional encodings
    embed_dim: int = 256  # Size of embedding matrix
    theta: float = 10000.0

    # Transformer Blocks
    num_embed_blocks: int = 3  # MLP layers added to embeddings
    num_blocks: int = 8  # Number of transformer blocks
    num_heads: int = 8
    hidden_dim: int = 512

    # Shared Layer Normalization
    eps: float = 1e-8
    dropout: float = 0.1
    bias: bool = False


@dataclass
class SamplerConfig(BaseConfig):
    """Configuration for sampling strategies in generation."""

    pad_id: Optional[int] = None
    top_k: int = 50
    top_p: float = 0.9
    temperature: float = 0.8
    repetition_penalty: float = 1.2
    greedy: bool = False  # Greedy decoding mode
    verbose: bool = False  # Enable debug mode


@dataclass
class GeneratorConfig(BaseConfig):
    """Configuration for text generation."""

    # Model
    runtime: RuntimeConfig
    state: MiniState
    sampler: MiniSampler

    # Tokenizer
    processor: SentencePieceProcessor  # Tokenizer
    pre_tokenizer: ClassVar[re.Pattern] = DEFAULT_PRETOKENIZER
