"""
Copyright © 2023 Austin Berrio
Module: mini.transformer.trainer
Description: Trainer for the MiniTransformer model.
"""

import random
from dataclasses import dataclass
from typing import Optional

import torch
from sentencepiece import SentencePieceProcessor
from torch import device, nn, optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from mini.data.set import MiniDataset
from mini.transformer.checkpoint import MiniCheckpoint
from mini.transformer.model import MiniTransformer, TransformerConfig


class MiniTrainer:
    def __init__(
        self,
        path: str,
        dataset: MiniDataset,
        processor: SentencePieceProcessor,
        config: TransformerConfig,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        criterion: nn.Module,
        checkpoint: MiniCheckpoint,
        device: device,
    ):
        self.path = path
        self.dataset = dataset
        self.processor = processor
        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.checkpoint = checkpoint
        self.device = device
        self.model = MiniTransformer(config).to(device)
        self.pad_id = self.processor.pad_id()
        if self.pad_id < 0:
            self.pad_id = 0

    def device() -> torch.device:
        # Automatically detect the physical device
        device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return device

    def seed(z: int) -> None:
        """Sets the random seed for reproducibility."""
        random.seed(z)
        torch.manual_seed(z)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(z)

    def log_batch(
        self,
        epoch: int,
        num_epochs: int,
        batch_idx: int,
        loss: torch.Tensor,
    ):
        """Logs the loss for each batch in an epoch."""
        perplexity = (
            torch.exp(loss).item()
            if isinstance(loss, torch.Tensor)
            else torch.exp(loss)
        )
        print(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"[{batch_idx}/{len(self.dataset)}] "
            f"[loss] {loss.item():.6f}, "
            f"[perplexity] {perplexity:.6f}, "
        )

    def log_epoch(self, epoch: int, total_loss: int):
        """Logs the epoch loss and learning rate."""
        print(
            f"[epoch completed] {epoch+1}, "
            f"[total loss] {total_loss:.4f}, "
            f"[avg loss] {total_loss / len(self.dataset):.4f}, "
            f"[lr] {self.scheduler.get_last_lr()[0]:.8f}, "
            f"[perplexity] {torch.exp(total_loss / len(self.dataset)).item():.6f} "
        )

    def train(
        self,
        num_epochs: int = 10,
        save_every: int = 10,
        grad_accum_steps: int = 1,
        verbose: bool = False,
    ):
        # Set the device type for autocast
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the model from the checkpoint
        self.model = self.checkpoint.load(config=self.config, device=self.device)

        # Set the module to train mode
        self.model.train()

        for epoch in range(num_epochs):
            total_loss = 0
            self.optimizer.zero_grad()

            for batch_idx, (x, y) in enumerate(self.dataset):
                x, y = x.to(device), y.to(device)
                mask = (x != 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)

                with torch.amp.autocast(device_type=device_type):
                    logits = self.model(x, mask)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss = loss / grad_accum_steps  # Normalize loss

                loss.backward()

                if (batch_idx + 1) % grad_accum_steps == 0 or batch_idx == len(
                    self.dataset
                ) - 1:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if verbose:
                    self.log_batch(epoch, num_epochs, batch_idx, loss)

            self.scheduler.step()  # Now updates LR per epoch
            total_loss += loss.item() * grad_accum_steps  # Re-scale
            self.log_epoch(epoch, total_loss)

            # Save the model periodically
            if (epoch + 1) % save_every == 0:
                self.checkpoint.save(self.model)
