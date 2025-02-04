"""
Copyright © 2023 Austin Berrio
Module: mini.transformer.manager
Description: Defines classes for managing optimizers, schedulers, and criteria in the transformer model.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler

from mini.common.logger import get_logger


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
    t_max: int = 50
    eta_min: float = 1e-6
    start_factor: float = 0.1
    total_iters: int = 50

    def get_keys(self) -> Dict[str, Any]:
        """Default scheduler parameters for common schedulers."""
        return {
            "step": {"step_size", "gamma"},
            "cosine": {"t_max", "eta_min"},
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


class MiniManager:
    def __init__(
        self,
        optimizer: Optional[OptimizerConfig] = None,
        scheduler: Optional[SchedulerConfig] = None,
        criterion: Optional[CriterionConfig] = None,
        verbose: bool = False,
    ):
        self.logger = get_logger(
            name=self.__class__.__name__,
            level=logging.DEBUG if verbose else logging.INFO,
        )

        self.optimizer_config = optimizer if optimizer else OptimizerConfig()
        self.scheduler_config = scheduler if scheduler else SchedulerConfig()
        self.criterion_config = criterion if criterion else CriterionConfig()

    def optimize(self, model: nn.Module) -> optim.Optimizer:
        """Creates an optimizer based on the given configuration."""
        optimizer_type = self.optimizer_config.type.lower()
        self.logger.info(f"Using optimizer: {optimizer_type}")

        optimizer_params = self.optimizer_config.get_params(optimizer_type)
        model_params = model.parameters(self.optimizer_config.recurse)
        if optimizer_type == "adam":
            return optim.Adam(model_params, **optimizer_params)
        elif optimizer_type == "adamw":
            return optim.AdamW(model_params, **optimizer_params)
        elif optimizer_type == "sgd":
            return optim.SGD(model_params, **optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    def schedule(self, optimizer: optim.Optimizer) -> Optional[LRScheduler]:
        """Creates a learning rate scheduler based on configuration."""
        scheduler_type = self.scheduler_config.type.lower()

        # **Handle Optional Scheduler**
        if scheduler_type == "none":
            self.logger.info("No learning rate scheduler is being used.")
            return None  # Early exit, no scheduler applied

        self.logger.info(f"Using scheduler: {scheduler_type}")

        scheduler_params = self.scheduler_config.get_params(scheduler_type)
        if scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
        elif scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **scheduler_params
            )
        elif scheduler_type == "linear":
            return torch.optim.lr_scheduler.LinearLR(optimizer, **scheduler_params)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    def criterion(self) -> nn.Module:
        """Creates a loss function based on the given criterion type."""
        criterion_type = self.criterion_config.type.lower()
        self.logger.info(f"Using criterion: {criterion_type}")

        criterion_params = self.criterion_config.get_params(criterion_type)
        if criterion_type == "cross_entropy":
            return nn.CrossEntropyLoss(**criterion_params)
        elif criterion_type == "mse":
            return nn.MSELoss(**criterion_params)
        elif criterion_type == "mae":
            return nn.L1Loss(**criterion_params)
        else:
            raise ValueError(f"Unsupported criterion: {criterion_type}")
