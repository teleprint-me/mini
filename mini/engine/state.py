"""
Copyright © 2023 Austin Berrio
Module: mini.engine.state
Description: Utility class for saving and loading checkpoints.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler

from mini.common.logger import get_logger
from mini.config import ConfigRuntime, ConfigTransformer
from mini.engine.optimizer_manager import EngineOptimizerManager
from mini.models.valerie import ModelValerie


@dataclass
class EngineState:
    """Manages training state, including model, optimizer, scheduler, and criterion."""

    path: str
    config: ConfigTransformer
    manager: EngineOptimizerManager
    runtime: ConfigRuntime
    verbose: bool = False

    checkpoint: Dict[str, Any] = field(default_factory=dict)
    model: Optional[nn.Module] = None
    optimizer: Optional[optim.Optimizer] = None
    scheduler: Optional[LRScheduler] = None
    criterion: Optional[nn.Module] = None

    def __post_init__(self):
        """Initializes state by loading from checkpoint (if exists) or setting up new components."""
        self.logger = get_logger(
            name=self.__class__.__name__,
            level=logging.DEBUG if self.verbose else logging.INFO,
        )

    def _load_checkpoint(self) -> None:
        """Loads checkpoint from disk if it exists, otherwise initializes a new checkpoint."""
        if os.path.exists(self.path):
            try:
                self.logger.info(f"Loading checkpoint from {self.path}")
                self.checkpoint = torch.load(self.path)
            except (torch.TorchScriptException, RuntimeError, IOError) as e:
                self.logger.error(f"Failed to load checkpoint: {e}")
                self.checkpoint = {}  # Start fresh if checkpoint is corrupt
        else:
            self.logger.info(f"No checkpoint found. Creating new state at {self.path}")
            self.checkpoint = {}

    def _load_model(self) -> None:
        """Loads model from checkpoint if available, otherwise initializes a new model."""
        if "model_config" in self.checkpoint:
            self.config = ConfigTransformer(**self.checkpoint["model_config"])

        self.model = ModelValerie(self.config)

        if "model_state" in self.checkpoint:
            self.model.load_state_dict(self.checkpoint["model_state"])
            self.logger.info("Model state loaded from checkpoint.")

        # Move model to the correct device using runtime information
        self.model.to(self.runtime.device_type)
        self.logger.info(f"Model moved to device: {self.runtime.device_name}")

    def _load_optimizer(self) -> None:
        """Loads optimizer and restores state if available."""
        self.optimizer = self.manager.create_optimizer(self.model)
        if "optimizer_state" in self.checkpoint:
            self.optimizer.load_state_dict(self.checkpoint["optimizer_state"])
            self.logger.info("Optimizer state loaded from checkpoint.")

    def _load_scheduler(self) -> None:
        """Loads scheduler and restores state if available."""
        self.scheduler = self.manager.create_scheduler(self.optimizer)

        if self.scheduler is None:
            self.logger.info("No scheduler is being used.")
            return  # Early exit if scheduler is disabled

        if "scheduler_state" in self.checkpoint:
            self.scheduler.load_state_dict(self.checkpoint["scheduler_state"])
            self.logger.info("Scheduler state loaded from checkpoint.")

    def _load_criterion(self) -> None:
        """Initializes criterion (loss function)."""
        self.criterion = self.manager.create_criterion()
        self.logger.info(f"Criterion: {self.criterion}")

    def load(self, training_mode: bool = False) -> None:
        """Loads training state from checkpoint or initializes new components."""
        self.logger.debug("Loading model state...")

        # Load model parameters for inference
        self._load_checkpoint()
        self._load_model()

        # Load optimization parameters for training
        if training_mode:
            self._load_optimizer()
            self._load_scheduler()
            self._load_criterion()

        # Log final loaded state
        self.logger.info(f"Model Configuration: {self.config}")

    def save(self) -> None:
        """Saves training state (model, optimizer, scheduler) to a checkpoint file."""
        if self.model is None:
            self.logger.error("Model is not initialized, cannot save checkpoint.")
            raise RuntimeError("Model is not initialized, cannot save checkpoint.")

        checkpoint = {
            "model_config": self.config.as_dict(),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "criterion_type": self.manager.criterion_config.type,
        }

        try:
            torch.save(checkpoint, self.path)
            self.logger.info(f"Checkpoint saved successfully to {self.path}")
        except IOError as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            raise
