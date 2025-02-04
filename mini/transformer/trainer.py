"""
Copyright © 2023 Austin Berrio
Module: mini.transformer.trainer
Description: Trainer for the MiniTransformer model.
"""

import logging
from typing import Union

import torch
from sentencepiece import SentencePieceProcessor

from mini.common.logger import get_logger
from mini.data.set import MiniDataset
from mini.transformer.state import MiniState


class MiniTrainer:
    def __init__(
        self,
        processor: SentencePieceProcessor,
        dataset: MiniDataset,
        state: MiniState,
        verbose: bool = False,
    ):
        self.processor = processor
        self.dataset = dataset
        self.state = state
        self.pad_id = max(self.processor.pad_id(), 0)  # Ensure pad_id is non-negative

        log_level = logging.DEBUG if verbose else logging.INFO
        self.logger = get_logger(name=self.__class__.__name__, level=log_level)

    def log_batch(
        self,
        epoch: int,
        num_epochs: int,
        batch: int,
        loss: torch.Tensor,
    ):
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
        self.logger.info(
            f"[Epoch: {epoch+1}/{num_epochs}] "
            f"[Total Loss: {total_loss:.4f}] "
            f"[Avg Loss: {average_loss:.4f}] "
            f"[LR: {self.state.scheduler.get_last_lr()[0]:.8f}] "
            f"[Perplexity: {self.perplexity(average_loss):.6f}]"
        )

    def average_loss(self, x: Union[float, torch.Tensor]) -> float:
        if isinstance(x, torch.Tensor):
            x = x.item()
        return x / len(self.dataset)

    def perplexity(self, x: Union[float, torch.Tensor]) -> float:
        if isinstance(x, float):
            x = torch.tensor(x)
        return torch.exp(x).item()

    def mask(self, x: torch.Tensor) -> torch.Tensor:
        """Creates a mask for the input tensor to ignore pad tokens."""
        # Mask set to (B, 1, T, T) to mask pad id
        return (
            (x != self.pad_id).unsqueeze(1).unsqueeze(2).expand(-1, -1, x.size(1), -1)
        )

    def train(
        self,
        num_epochs: int = 10,
        save_every: int = 10,
        grad_accum_steps: int = 1,
    ):
        """Trains the model with gradient accumulation support."""
        self.logger.info("Starting training...")

        self.state.load()
        device = self.state.runtime.device_type

        model = self.state.model.to(device)
        model.train()

        optimizer = self.state.optimizer
        scheduler = self.state.scheduler
        criterion = self.state.criterion

        for epoch in range(num_epochs):
            total_loss = 0
            for batch, (x, y) in enumerate(self.dataset):
                optimizer.zero_grad()
                x, y = x.to(device), y.to(device)

                logits = model(x, self.mask(x))
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / grad_accum_steps  # Normalize loss for accumulation

                loss.backward(retain_graph=True)
                # optimizer.step()  # temporarily block this for debugging
                total_loss += loss.item() * grad_accum_steps

                # Debug: Check if weights are updating
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        before = param.clone().detach()
                        optimizer.step()  # Apply optimizer updates
                        after = param.clone().detach()
                        diff = torch.norm(after - before).item()
                        assert diff > 0, f"Weight {name} is not updating!"
                        break

                self.log_batch(epoch, num_epochs, batch, loss)

            scheduler.step()  # Step LR scheduler **once per epoch**
            self.log_epoch(epoch, num_epochs, total_loss)

            # Save model periodically
            if (epoch + 1) % save_every == 0:
                self.state.save()

        self.logger.info("Training complete!")
