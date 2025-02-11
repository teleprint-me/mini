"""
Copyright © 2023 Austin Berrio
Module: mini.config.device
Description: Manages runtime-specific settings like device handling and seeding.
"""

import random
from dataclasses import dataclass, field

import torch

from mini.config.base import ConfigBase


@dataclass
class ConfigDevice(ConfigBase):
    """Manages runtime-specific settings like device handling and seeding."""

    seed: int = 42  # Random seed for reproducibility
    dname: str = field(init=False, default="cpu")  # Device name
    dtype: torch.dtype = field(init=False, default=torch.float32)  # Tensor dtype

    def __post_init__(self):
        """Initialize device based on availability."""
        self.set_device()

    @property
    def device(self) -> torch.device:
        """Returns the best available device as a `torch.device` object."""
        return torch.device(self.dname)

    def set_device(self) -> None:
        """Sets the default device and dtype, and clears CUDA cache if available."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.dname = "cuda"
        torch.set_default_dtype(self.dtype)
        self.set_seed()

    def set_seed(self) -> None:
        """Sets the random seed for reproducibility."""
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
