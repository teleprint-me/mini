"""
Copyright © 2023 Austin Berrio
Module: mini.config.runtime
Description: Manages runtime-specific settings like device handling and seeding.
"""

import random
from dataclasses import dataclass, field

import torch

from mini.config.base import ConfigBase


@dataclass
class ConfigRuntime(ConfigBase):
    """Manages runtime-specific settings like device handling and seeding."""

    seed: int = 42
    device_name: str = field(init=False, default="cpu")

    def __post_init__(self):
        """Initialize device based on availability."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.device_name = "cuda"

    @property
    def device_type(self) -> torch.device:
        """Returns the best available device as a `torch.device` object."""
        return torch.device(self.device_name)

    def set_seed(self) -> None:
        """Sets the random seed for reproducibility."""
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
