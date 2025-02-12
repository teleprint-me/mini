"""
Copyright © 2023 Austin Berrio
Module: mini.config.trainer
Description: Configuration for optimizer, scheduler, and criterion.
"""

from dataclasses import dataclass
from typing import Any, Dict

from mini.config.base import ConfigBase


@dataclass
class ConfigOptimizerManager(ConfigBase):
    """Base class for optimizer, scheduler, and criterion configs."""

    type: str

    def get_keys(self) -> Dict[str, Any]:
        """Returns allowed parameters for different object types."""
        raise ValueError(f"{self.__class__.__name__} must define `get_keys()`.")

    def get_params(self, key: str) -> Dict[str, Any]:
        """Filters parameters relevant to the specified object type."""
        key = key.lower()
        params = self.as_dict()
        keys = self.get_keys()
        if key not in keys:
            raise ValueError(f"Unsupported type: {key}")
        return {k: v for k, v in params.items() if k in keys[key]}


@dataclass
class ConfigOptimizer(ConfigOptimizerManager):
    """Configuration for optimizers like Adam, AdamW, and SGD."""

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
class ConfigScheduler(ConfigOptimizerManager):
    """Configuration for learning rate schedulers."""

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
class ConfigCriterion(ConfigOptimizerManager):
    """Configuration for loss functions."""

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
