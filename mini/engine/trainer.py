"""
Copyright © 2023 Austin Berrio
Module: mini.engine.trainer
Description: Trainer for the MiniTransformer model.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Union

import torch
from sentencepiece import SentencePieceProcessor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from mini.common.logger import get_logger
from mini.data.loader import DatasetLoader
from mini.engine.state import EngineState


@dataclass
class EngineTrainer:
    """Trainer for the MiniTransformer model."""

    processor: SentencePieceProcessor
    dataset: DatasetLoader
    state: EngineState
    verbose: bool = False

    def __post_init__(self):
        self.pad_id = max(self.processor.pad_id(), 0)  # Ensure pad_id is non-negative
        self.logger = get_logger(
            name=self.__class__.__name__,
            level=logging.DEBUG if self.verbose else logging.INFO,
        )

    # === 🔥 Convenience Properties === #
    @property
    def device(self) -> torch.device:
        return self.state.config.device

    @property
    def model(self) -> torch.nn.Module:
        return self.state.model

    @property
    def optimizer(self) -> Optimizer:
        return self.state.optimizer

    @property
    def scheduler(self) -> Optional[LRScheduler]:
        return self.state.scheduler

    @property
    def loss(self) -> torch.nn.Module:
        return self.state.criterion

    def load(self) -> None:
        """Loads state and sets model to training mode."""
        self.state.load(training_mode=True)
        self.model.train()

    def save(self) -> None:
        """Saves current model state."""
        self.state.save()

    # === 🔥 Training Methods === #
    def train(
        self,
        num_epochs: int = 10,
        save_every: int = 10,
        grad_accum_steps: int = 1,
    ):
        """Trains the model with gradient accumulation support."""
        self.logger.info("Starting training...")
        self.load()
        self.log_parameters()

        for epoch in range(num_epochs):
            total_loss = 0
            for batch, (x, y) in enumerate(self.dataset):
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.loss(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / grad_accum_steps  # Normalize loss per step

                loss.backward()

                if (batch + 1) % grad_accum_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                total_loss += loss.item() * grad_accum_steps

                self.log_batch(epoch, num_epochs, batch, loss)

            if self.scheduler is not None:
                self.scheduler.step()  # Step LR scheduler **once per epoch**
            self.log_epoch(epoch, num_epochs, total_loss)

            # Save model periodically
            if (epoch + 1) % save_every == 0:
                self.save()

        self.logger.info("Training complete!")

    # === 🔥 Logging & Utilities === #
    def log_parameters(self):
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model has {num_params:,} learnable parameters.")

    def log_batch(self, epoch: int, num_epochs: int, batch: int, loss: torch.Tensor):
        """Logs loss & perplexity for each batch."""
        self.logger.debug(
            f"[Epoch: {epoch+1}/{num_epochs}] "
            f"[Batch: {batch}/{len(self.dataset)}] "
            f"[Loss: {loss.item():.6f}] "
            f"[Perplexity: {self.perplexity(loss):.6f}]"
        )

    def log_epoch(self, epoch: int, num_epochs: int, total_loss: float):
        """Logs total epoch loss, learning rate, and perplexity."""
        average_loss = self.average_loss(total_loss)
        lr = self.optimizer.param_groups[0]["lr"]

        self.logger.info(
            f"[Epoch: {epoch+1}/{num_epochs}] "
            f"[Total Loss: {total_loss:.4f}] "
            f"[Avg Loss: {average_loss:.4f}] "
            f"[LR: {lr:.8f}] "
            f"[Perplexity: {self.perplexity(average_loss):.6f}]"
        )

    def average_loss(self, x: Union[float, torch.Tensor]) -> float:
        if isinstance(x, torch.Tensor):
            x = x.item()
        return x / len(self.dataset)

    def perplexity(self, x: Union[float, torch.Tensor]) -> float:
        """Computes perplexity, ensuring loss is non-negative."""
        if isinstance(x, float):
            x = torch.tensor(x)
        x = torch.clamp(x, min=0)
        return torch.exp(x).item()
