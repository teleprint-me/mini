"""
Copyright © 2023 Austin Berrio
Script: mini.transformer.train
Description: Simple pre-training loop for text-to-text generation.
"""

import argparse
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from sentencepiece import SentencePieceProcessor
from torch.optim.lr_scheduler import LRScheduler

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
):
    # Load the model and optimizer if a checkpoint exists
    model, optimizer = load_checkpoint(model_path, model, optimizer)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataset):
            x, y = x.to(device), y.to(device)

            # Create mask where PAD (0) tokens are ignored
            mask = (x != 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)

            optimizer.zero_grad()
            logits = model(x, mask)  # (Batch, Seq Len, Vocab Size)

            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            print(
                f"[Epoch {epoch+1}/{num_epochs}] [{batch_idx}/{len(dataset)}] Loss: {loss.item():.4f}"
            )

        if (epoch + 1) % scheduler.step_size == 0:
            scheduler.step()
        print(
            f"Epoch {epoch+1} Completed | Avg Loss: {total_loss / len(dataset):.4f} | LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Save the model and optimizer periodically
        if (epoch + 1) % save_every == 0:
            save_checkpoint(model_path, model, optimizer)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MiniTransformer")
    parser.add_argument(
        "--processor", required=True, help="Path to SentencePiece tokenizer model."
    )
    parser.add_argument(
        "--schema", required=True, help="Path to the schema file for the dataset."
    )
    parser.add_argument(
        "--dataset", required=True, help="Path to a raw plaintext file."
    )
    parser.add_argument(
        "--model", required=True, help="Path to save or load the model."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (Default: 42).",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=512,
        help="Embedding dimension size (Default: 512).",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads (Default: 8).",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=64,
        help="Head dimension size (Default: 64).",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=8,
        help="Number of transformer layers (Default: 8).",
    )
    parser.add_argument(
        "--ff-dim",
        type=int,
        default=512,
        help="Feed-forward network dimension (Default: 512).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Maximum sequence length (Default: 512).",
    )
    parser.add_argument(
        "--rope-theta",
        type=float,
        default=10000.0,
        help="Theta value for RoPE positional encoding (Default: 10000.0).",
    )
    parser.add_argument(
        "--bias",
        type=bool,
        default=False,
        help="Use bias in the feed-forward network (Default: False).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training (Default: 8).",
    )
    parser.add_argument(
        "--batch-stride",
        type=int,
        default=64,
        help="Stride for batching the dataset (Default: 64).",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs (Default: 10).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save model every N epochs (Default: 10).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Epsilon value for numerical stability (Default: 1e-8).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
        help="Weight decay for regularization (Default: 0.0001).",
    )
    parser.add_argument(
        "--amsgrad",
        action="store_true",
        help="Use AMSGrad for optimizer (Default: False).",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=10,
        help="Learning rate scheduler step size (Default: 10).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.8,
        help="Learning rate scheduler gamma (Default: 0.8).",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="Learning rate (Default: 5e-4)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

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
        max_seq_len=args.n_seq_len,
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
        theta=args.theta,
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
    )
    print("Training complete!")
