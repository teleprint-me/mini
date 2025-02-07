"""
Copyright © 2023 Austin Berrio
Module: mini.engine.optimizer
Description: Defines classes for managing optimizers, schedulers, and criteria in the transformer model.
"""

import logging
from typing import Optional

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler

from mini.common.logger import get_logger
from mini.config import MiniConfigCriterion, MiniConfigOptimizer, MiniConfigScheduler


class MiniEngineOptimizer:
    def __init__(
        self,
        optimizer: Optional[MiniConfigOptimizer] = None,
        scheduler: Optional[MiniConfigScheduler] = None,
        criterion: Optional[MiniConfigCriterion] = None,
        verbose: bool = False,
    ):
        self.logger = get_logger(
            name=self.__class__.__name__,
            level=logging.DEBUG if verbose else logging.INFO,
        )

        self.optimizer_config = optimizer if optimizer else MiniConfigOptimizer()
        self.scheduler_config = scheduler if scheduler else MiniConfigScheduler()
        self.criterion_config = criterion if criterion else MiniConfigCriterion()

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
