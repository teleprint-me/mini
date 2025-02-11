"""
Copyright © 2023 Austin Berrio
Module: mini.config.transformer
Description: Configuration settings for the transformer model.
NOTE: ConfigDevice inherits from ConfigBase.
"""

from dataclasses import dataclass

from mini.config.device import ConfigDevice


@dataclass
class ConfigTransformer(ConfigDevice):
    """Configuration for transformer model architecture."""

    # Encoder and Embedding
    pad_id: int = -1  # Handle padding in embeddings
    vocab_size: int = 32000  # Number of unique tokens in the vocabulary
    max_seq_len: int = 128  # Max positional encodings
    seq_len: int = 6  # Sequence length
    embed_dim: int = 256  # Size of embedding matrix
    rope_theta: float = 10000.0  # RoPE scaling factor

    # Transformer Blocks
    num_mlp_layers: int = 3  # MLP layers added to embeddings
    num_layers: int = 8  # Number of transformer blocks
    num_heads: int = 8  # Number of attention heads per block
    ff_dim: int = 512  # Hidden dim for feed-forward network
    ff_mult: float = 4.0  # FFN multiplier (default 4.0, standard for transformers)

    # Shared Layer Normalization
    eps: float = 1e-8
    dropout: float = 0.1
    bias: bool = False

    def __post_init__(self):
        """Initializes model parameters and ensures correctness."""
        # Set the pad id to 0 if it's unset
        self.pad_id = max(self.pad_id, 0)

        # Compute derived parameters
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = 1 / (self.head_dim**0.5)  # Corrected scale calculation
        self.ff_dim = int(self.embed_dim * self.ff_mult)  # Compute FFN hidden size

        # Validate parameter correctness
        self.__assert_init__()

        # Initialize device based on availability
        self.set_device()

    def __assert_init__(self):
        """Ensures model parameters are correctly initialized."""
        assert (
            self.embed_dim % 2 == 0
        ), f"Embedding dim must be even for sin/cos. Given: {self.embed_dim}"
        assert (
            self.embed_dim % self.num_heads == 0
        ), f"Embedding dim must be divisible by num_heads. Given: {self.embed_dim}, num_heads: {self.num_heads}"
        assert (
            self.head_dim == self.embed_dim // self.num_heads
        ), f"Head dim mismatch: {self.head_dim} != {self.embed_dim // self.num_heads}"
        assert self.scale == 1 / (
            self.head_dim**0.5
        ), f"Scale mismatch: {self.scale} != 1 / sqrt({self.head_dim})"
