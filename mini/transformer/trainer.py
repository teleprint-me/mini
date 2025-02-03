"""
Copyright © 2023 Austin Berrio
Module: mini.transformer.trainer
Description: Trainer for the MiniTransformer model.
"""

import torch
from sentencepiece import SentencePieceProcessor
from torch import device, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from mini.data.set import MiniDataset
from mini.transformer.checkpoint import MiniCheckpoint
from mini.transformer.model import MiniTransformer, TransformerConfig


class MiniTrainer:
    def __init__(
        self,
        processor: SentencePieceProcessor,
        dataset: MiniDataset,
        model: MiniTransformer,
        config: TransformerConfig,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        criterion: nn.Module,
        checkpoint: MiniCheckpoint,
        device: device,
        verbose: bool = False,
    ):
        self.processor = processor
        self.dataset = dataset
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.checkpoint = checkpoint
        self.device = device
        self.verbose = verbose
        self.pad_id = max(self.processor.pad_id(), 0)  # Ensure pad_id is non-negative

    def log_batch(
        self, epoch: int, num_epochs: int, batch_idx: int, loss: torch.Tensor
    ):
        """Logs loss & perplexity for each batch."""
        perplexity = (
            torch.exp(loss).item()
            if isinstance(loss, torch.Tensor)
            else torch.exp(loss)
        )
        print(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"[{batch_idx}/{len(self.dataset)}] "
            f"[loss] {loss.item():.6f}, "
            f"[perplexity] {perplexity:.6f}"
        )

    def log_epoch(self, epoch: int, total_loss: float):
        """Logs total epoch loss, learning rate, and perplexity."""
        avg_loss = total_loss / len(self.dataset)
        print(
            f"[Epoch {epoch+1}] "
            f"[Total Loss: {total_loss:.4f}] "
            f"[Avg Loss: {avg_loss:.4f}] "
            f"[LR: {self.scheduler.get_last_lr()[0]:.8f}] "
            f"[Perplexity: {torch.exp(torch.tensor(avg_loss)).item():.6f}]"
        )

    def resume(self) -> None:
        """Load checkpoint, restoring optimizer/scheduler if available."""
        loaded = self.checkpoint.load()
        if loaded[0] is not None:
            self.model = loaded[0]
        if loaded[1] is not None:
            self.optimizer = loaded[1]

    def train(
        self,
        num_epochs: int = 10,
        save_every: int = 10,
        grad_accum_steps: int = 1,
    ):
        """Trains the model with gradient accumulation support."""
        if self.verbose:
            print("Starting training...")

        self.resume()

        # Ensure model is in train mode
        self.model.train()

        for epoch in range(num_epochs):
            total_loss = 0
            self.optimizer.zero_grad()

            for batch_idx, (x, y) in enumerate(self.dataset):
                x, y = x.to(self.device), y.to(self.device)
                mask = (x != self.pad_id).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)

                with torch.amp.autocast(device_type=self.device.type):
                    logits = self.model(x, mask)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss = loss / grad_accum_steps  # Normalize loss for accumulation

                loss.backward()
                total_loss += loss.item() * grad_accum_steps
                even_step = (batch_idx + 1) % grad_accum_steps == 0
                last_batch = batch_idx == len(self.dataset) - 1
                if even_step or last_batch:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if self.verbose:
                    self.log_batch(epoch, num_epochs, batch_idx, loss)

            self.scheduler.step()  # Step the LR scheduler **once per epoch**
            self.log_epoch(epoch, total_loss)

            # Save model periodically
            if (epoch + 1) % save_every == 0:
                self.checkpoint.save(self.model)

        if self.verbose:
            print("Training complete!")
