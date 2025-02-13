"""
Copyright © 2023 Austin Berrio
Module: mini.engine.optimizer_manager
Description: Defines classes for managing optimizers, schedulers, and criteria in the transformer model.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler

from mini.common.logger import get_logger
from mini.config.optimizer_manager import (
    ConfigCriterion,
    ConfigOptimizer,
    ConfigScheduler,
)


class EngineOptimizerManager:
    def __init__(
        self,
        config_optimizer: Optional[ConfigOptimizer] = None,
        config_scheduler: Optional[ConfigScheduler] = None,
        config_criterion: Optional[ConfigCriterion] = None,
        verbose: bool = False,
    ):
        self.config_optimizer = (
            config_optimizer if config_optimizer else ConfigOptimizer()
        )
        self.config_scheduler = (
            config_scheduler if config_scheduler else ConfigScheduler()
        )
        self.config_criterion = (
            config_criterion if config_criterion else ConfigCriterion()
        )
        self.verbose = verbose
        self.logger = get_logger(
            name=self.__class__.__name__,
            level=logging.DEBUG if self.verbose else logging.INFO,
        )

    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Creates an optimizer based on the given configuration."""
        optimizer_type = self.config_optimizer.type.lower()
        self.logger.info(f"Using optimizer: {optimizer_type}")

        optimizer_params = self.config_optimizer.get_params(optimizer_type)
        self.logger.debug(f"Optimizer parameters: {optimizer_params}")
        if optimizer_type == "adam":
            return optim.Adam(
                model.parameters(self.config_optimizer.recurse), **optimizer_params
            )
        elif optimizer_type == "adamw":
            return optim.AdamW(
                model.parameters(self.config_optimizer.recurse), **optimizer_params
            )
        elif optimizer_type == "sgd":
            return optim.SGD(
                model.parameters(self.config_optimizer.recurse), **optimizer_params
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    def create_scheduler(self, optimizer: optim.Optimizer) -> Optional[LRScheduler]:
        """Creates a learning rate scheduler based on configuration."""
        scheduler_type = self.config_scheduler.type.lower()

        # **Handle Optional Scheduler**
        if scheduler_type == "none":
            self.logger.info("No learning rate scheduler is being used.")
            return None  # Early exit, no scheduler applied

        self.logger.info(f"Using scheduler: {scheduler_type}")

        scheduler_params = self.config_scheduler.get_params(scheduler_type)
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

    def create_criterion(self) -> nn.Module:
        """Creates a loss function based on the given criterion type."""
        criterion_type = self.config_criterion.type.lower()
        self.logger.info(f"Using criterion: {criterion_type}")

        criterion_params = self.config_criterion.get_params(criterion_type)
        if criterion_type == "cross_entropy":
            return nn.CrossEntropyLoss(**criterion_params)
        elif criterion_type == "mse":
            return nn.MSELoss(**criterion_params)
        elif criterion_type == "mae":
            return nn.L1Loss(**criterion_params)
        else:
            raise ValueError(f"Unsupported criterion: {criterion_type}")
