"""
Module: mini.transformer.config
Description: Common configuration components for transformer models.
"""

import functools
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import regex as re
import torch

from mini.transformer.processor import SentencePieceProcessor
from mini.transformer.sampler import MiniSampler
from mini.transformer.state import MiniState

# Default GPT-2 style pre-tokenizer regex
DEFAULT_PRETOKENIZER = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    re.IGNORECASE,
)


@dataclass
class RuntimeConfig:
    """Manages runtime-specific settings like device handling and seeding."""

    seed: int = 42

    @functools.cached_property
    def device_name(self) -> str:
        """Returns the best available device name ('cuda' or 'cpu')."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            return "cuda"
        return "cpu"

    @functools.cached_property
    def device_type(self) -> torch.device:
        """Returns the best available device as a `torch.device` object."""
        return torch.device(self.device_name)

    def seed_all(self) -> None:
        """Sets the random seed for reproducibility."""
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)


@dataclass
class TransformerConfig:
    vocab_size: int = 32000
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 8
    ff_dim: int = 512
    max_seq_len: int = 128
    pad_id: int = -1
    dropout: float = 0.1
    eps: float = 1e-8
    theta: float = 10000.0
    bias: bool = False

    def as_dict(self) -> dict[str, any]:
        """Returns a dictionary representation of the config."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class GeneratorConfig:
    state: MiniState
    sampler: MiniSampler
    runtime: RuntimeConfig
    processor: SentencePieceProcessor
    pre_tokenizer: Optional[re.Pattern] = None


@dataclass
class SamplerConfig:
    top_k: int = 50
    top_p: float = 0.9
    temperature: float = 0.8
    repetition_penalty: float = 1.2
    greedy: bool = False  # Greedy decoding mode
    pad_id: Optional[int] = None
    verbose: bool = False  # Enable debug mode


@dataclass
class ManagerConfig:
    """Base class for optimizer, scheduler, and criterion configs."""

    type: str

    def as_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the config."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def get_keys(self) -> Dict[str, Any]:
        """Returns allowed parameters for different object types."""
        raise NotImplementedError("Subclasses must implement `get_keys()`.")

    def get_params(self, key: str) -> Dict[str, Any]:
        """Filters parameters relevant to the specified object type."""
        key = key.lower()
        params = self.as_dict()
        keys = self.get_keys()
        if key not in keys:
            raise ValueError(f"Unsupported type: {key}")
        return {k: v for k, v in params.items() if k in keys[key]}


@dataclass
class OptimizerConfig(ManagerConfig):
    type: str = "adamw"  # Default optimizer
    recurse: bool = True
    lr: float = 1e-3
    eps: float = 1e-8
    amsgrad: bool = False
    weight_decay: float = 0
    momentum: float = 0
    dampening: float = 0
    nesterov: bool = False

    def get_keys(self) -> Dict[str, Any]:
        """Default optimizer parameters for common optimizers."""
        return {
            "adam": {"lr", "eps", "weight_decay", "amsgrad"},
            "adamw": {"lr", "eps", "weight_decay", "amsgrad"},
            "sgd": {"lr", "weight_decay", "momentum", "dampening", "nesterov"},
        }


@dataclass
class SchedulerConfig(ManagerConfig):
    type: str = "step"  # Default scheduler
    step_size: int = 10
    gamma: float = 0.1
    T_max: int = 50
    eta_min: float = 1e-6
    start_factor: float = 0.1
    total_iters: int = 50

    def get_keys(self) -> Dict[str, Any]:
        """Default scheduler parameters for common schedulers."""
        return {
            "step": {"step_size", "gamma"},
            "cosine": {"T_max", "eta_min"},
            "linear": {"start_factor", "total_iters"},
        }


@dataclass
class CriterionConfig(ManagerConfig):
    type: str = "cross_entropy"  # Default loss function
    ignore_index: int = -1
    reduction: str = "mean"

    def get_keys(self) -> Dict[str, Any]:
        """Loss function parameters."""
        return {
            "cross_entropy": {"ignore_index", "reduction"},
            "mse": {"reduction"},
            "mae": {"reduction"},
        }
