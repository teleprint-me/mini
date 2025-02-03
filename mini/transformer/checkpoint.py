"""
Module: mini.transformer.checkpoint
Description: Utility class for saving and loading checkpoints.
"""

import os
from typing import Optional, Tuple

import torch
from torch import nn, optim

from mini.transformer.model import MiniTransformer, TransformerConfig


class MiniCheckpoint:
    """
    Class: MiniCheckpoint
    Description: Utility class for saving and loading checkpoints.
    """

    def __init__(
        self,
        path: str,
        optimizer: Optional[optim.Optimizer] = None,
        verbose: bool = False,
    ):
        """Initializes the MiniCheckpoint object."""
        self.path = path
        self.optimizer = optimizer
        self.verbose = verbose

    def save(self, model: nn.Module) -> None:
        """Saves the model and optimizer state to a checkpoint file."""
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
        }
        torch.save(checkpoint, self.path)
        if self.verbose:
            print(f"Saved model to {self.path}")

    def load(
        self,
        config: Optional[TransformerConfig] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[nn.Module, Optional[optim.Optimizer]]:
        """Loads a checkpoint and returns the model (and optimizer, if applicable)."""
        if self.verbose:
            print(f"Loading checkpoint from {self.path}")

        # Load checkpoint if it exists
        checkpoint = torch.load(self.path) if os.path.exists(self.path) else {}

        # Load config dynamically
        if config is None and "model_config" in checkpoint:
            config = TransformerConfig(**checkpoint["model_config"])

        # Reconstruct model dynamically
        model = MiniTransformer(config).to(device)

        # Load model state
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])

        # Load optimizer state if available
        if self.optimizer is not None and "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        return (model, self.optimizer) if self.optimizer else (model, None)
