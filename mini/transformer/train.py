"""
Copyright © 2023 Austin Berrio
Script: mini.transformer.train
Description: Simple pre-training loop for text-to-text generation.
"""

import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from sentencepiece import SentencePieceProcessor
from torch.optim.lr_scheduler import LRScheduler

from mini.common.args import TransformerArgs
from mini.data.set import MiniDataset, MiniTextDataset
from mini.transformer.model import MiniTransformer, TransformerConfig


def load_checkpoint(
    model_path: str, model: nn.Module, optimizer: optim.Optimizer
) -> tuple[nn.Module, optim.Optimizer]:
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return model, optimizer


def save_checkpoint(model_path: str, model: nn.Module, optimizer: optim.Optimizer):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(checkpoint, model_path)
    print(f"Saved model to {model_path}")


def train(
    model_path: str,
    model: nn.Module,
    dataset: MiniDataset,
    optimizer: optim.Optimizer,
    scheduler: LRScheduler,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int = 10,
    save_every: int = 10,
    grad_accum_steps: int = 1,
    mixed_precision: str = "none",
):
    # Load the model and optimizer if a checkpoint exists
    model, optimizer = load_checkpoint(model_path, model, optimizer)
    model.train()

    # Mixed precision setup
    scaler = torch.cuda.amp.GradScaler() if mixed_precision == "fp16" else None

    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()

        for batch_idx, (x, y) in enumerate(dataset):
            x, y = x.to(device), y.to(device)

            # Create mask where PAD (0) tokens are ignored
            mask = (x != 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)

            # Forward pass (mixed precision if enabled)
            with (
                torch.cuda.amp.autocast(
                    dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float16
                )
                if mixed_precision in ["fp16", "bf16"]
                else torch.no_grad()
            ):
                logits = model(x, mask)  # (Batch, Seq Len, Vocab Size)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / grad_accum_steps  # Normalize loss for accumulation

            # Backward pass
            if mixed_precision == "fp16":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Accumulate gradients
            if (batch_idx + 1) % grad_accum_steps == 0 or batch_idx == len(dataset) - 1:
                if mixed_precision == "fp16":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()  # Adjust learning rate

            total_loss += loss.item() * grad_accum_steps  # Re-scale

            print(
                f"[Epoch {epoch+1}/{num_epochs}] [{batch_idx}/{len(dataset)}] Loss: {loss.item():.4f}"
            )

        print(
            f"Epoch {epoch+1} Completed | Avg Loss: {total_loss / len(dataset):.4f} | LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Save the model periodically
        if (epoch + 1) % save_every == 0:
            save_checkpoint(model_path, model, optimizer)

    print("Training complete!")


if __name__ == "__main__":
    args = TransformerArgs("Mini Training Tool").parse_args()

    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load tokenizer
    processor = SentencePieceProcessor(model_file=args.processor)

    # Automatically detect the physical device
    device = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Setup PyTorch Dataset & DataLoader
    dataset = MiniTextDataset(
        file_path=args.dataset,
        processor=processor,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        stride=args.batch_stride,
    )

    # Model & Training Setup
    vocab_size = processor.vocab_size()
    pad_id = processor.pad_id()
    if pad_id < 0:
        pad_id = 0

    # Load Transformer Config
    config = TransformerConfig(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        max_seq_len=args.max_seq_len,
        pad_id=pad_id,
        theta=args.rope_theta,
        bias=args.bias,
    )

    # Initialize Model & Optimizer
    model = MiniTransformer(config).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        eps=args.eps,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)  # Ignore pad token

    print("Starting training...")
    train(
        model_path=args.model,
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        num_epochs=args.num_epochs,
        save_every=args.save_every,
        grad_accum_steps=args.grad_accum_steps,
        mixed_precision=args.mixed_precision,
    )
    print("Training complete!")
