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

    # Model Architecture
    architecture: str = "misty"  # Default architecture is "misty"

    # Encoder and Embedding
    pad_id: int = -1  # Handle padding in embeddings
    vocab_size: int = 32000  # Number of unique tokens in the vocabulary
    max_seq_len: int = 128  # Max positional encodings
    embed_dim: int = 256  # Size of embedding matrix
    rope_theta: float = 10000.0  # RoPE scaling factor

    # Transformer Blocks
    num_mlp_layers: int = 3  # MLP layers added to embeddings
    num_layers: int = 8  # Number of transformer blocks
    num_heads: int = 8  # Number of attention heads per block
    ff_dim: int = 512  # Hidden dim for feed-forward network
    ff_mult: float = 4.0  # FFN multiplier (default 4.0, standard for transformers)
    mask_type: str = "causal"  # Mask type for attention (causal, bidirectional)

    # Shared Layer Normalization
    eps: float = 1e-8
    dropout: float = 0.1
    bias: bool = False

    def __post_init__(self):
        """Initializes model parameters and ensures correctness."""
        # Ensure pad_id is non-negative.
        self.pad_id = max(self.pad_id, 0)

        # Compute derived parameters
        self.head_dim = self.embed_dim // self.num_heads  # Attention head dimension
        self.scale = self.head_dim**-0.5  # Scale factor for attention
        self.hidden_dim = int(self.ff_dim * self.ff_mult)  # Compute FFN hidden size

        # Validate parameter correctness
        self.__assert_init__()

        # Initialize device based on availability
        self.set_device()

    def __assert_init__(self):
        """Ensures model parameters are correctly initialized."""
        assert (
            self.embed_dim % 2 == 0
        ), f"Embedding dim must be even for sin/cos encoding. Given: {self.embed_dim}"

        assert self.embed_dim % self.num_heads == 0, (
            f"Embedding dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads}). "
            f"Suggested: Set embed_dim to a multiple of num_heads."
        )

        assert self.head_dim == self.embed_dim // self.num_heads, (
            f"Head dim mismatch: {self.head_dim} != {self.embed_dim // self.num_heads}. "
            f"Check num_heads and embed_dim configuration."
        )

        assert self.scale == self.head_dim**-0.5, (
            f"Scale mismatch: {self.scale} != {self.head_dim**-0.5}. "
            f"Ensure self.head_dim = embed_dim / num_heads is correct."
        )
