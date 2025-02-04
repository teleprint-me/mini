"""
Copyright © 2023 Austin Berrio
Module: mini.transformer.state
Description: Utility class for saving and loading checkpoints.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler

from mini.transformer.manager import MiniManager
from mini.transformer.model import MiniConfig, MiniRuntime, MiniTransformer


@dataclass
class MiniState:
    """Manages training state, including model, optimizer, scheduler, and criterion."""

    path: str
    config: MiniConfig
    manager: MiniManager
    runtime: MiniRuntime
    verbose: bool = False

    checkpoint: Dict[str, Any] = field(default_factory=dict)
    model: Optional[nn.Module] = None
    optimizer: Optional[optim.Optimizer] = None
    scheduler: Optional[LRScheduler] = None
    criterion: Optional[nn.Module] = None

    def __post_init__(self):
        """Initializes state by loading from checkpoint (if exists) or setting up new components."""
        self.load()

    def _load_checkpoint(self) -> None:
        if os.path.exists(self.path):
            if self.verbose:
                print(f"Loading checkpoint from {self.path}")
            self.checkpoint = torch.load(self.path)
        else:
            if self.verbose:
                print(f"Creating checkpoint for {self.path}")
            self.checkpoint = {}

    def _load_model(self) -> None:
        # **Step 1: Load Model**
        if "model_config" in self.checkpoint:
            self.config = MiniConfig(**self.checkpoint["model_config"])

        self.model = MiniTransformer(self.config)
        if "model_state" in self.checkpoint:
            self.model.load_state_dict(self.checkpoint["model_state"])
            if self.verbose:
                print("Model state loaded from checkpoint.")

    def _load_optimizer(self) -> None:
        # **Step 2: Create & Restore Optimizer**
        self.optimizer = self.manager.optimize(self.model)
        if "optimizer_state" in self.checkpoint:
            self.optimizer.load_state_dict(self.checkpoint["optimizer_state"])
            if self.verbose:
                print("Optimizer state loaded from checkpoint.")

    def _load_scheduler(self) -> None:
        # **Step 3: Create & Restore Scheduler**
        self.scheduler = self.manager.schedule(self.optimizer)
        if "scheduler_state" in self.checkpoint:
            self.scheduler.load_state_dict(self.checkpoint["scheduler_state"])
            if self.verbose:
                print("Scheduler state loaded from checkpoint.")

    def _load_criterion(self) -> None:
        # **Step 4: Initialize Criterion**
        self.criterion = self.manager.criterion()

    def load(self) -> None:
        """Loads training state from checkpoint or initializes new components."""
        self._load_checkpoint()
        self._load_model()
        self._load_optimizer()
        self._load_scheduler()
        self._load_criterion()

    def save(self) -> None:
        """Saves training state (model, optimizer, scheduler) to a checkpoint file."""
        if self.model is None:
            raise RuntimeError("Model is not initialized, cannot save checkpoint.")

        checkpoint = {
            "model_config": self.config.as_dict(),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "criterion_type": self.manager.criterion_config.type,
        }
        torch.save(checkpoint, self.path)
        if self.verbose:
            print(f"Checkpoint saved to {self.path}")
