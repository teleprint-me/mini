"""
Script: mini.cli.train
Description: Simple training loop for text-to-text generation.
"""

import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from sentencepiece import SentencePieceProcessor
from torch.utils.data import DataLoader, Dataset

from mini.common.json import JsonUtils
from mini.model.transformer import MiniTransformer


class DummyDataset(Dataset):
    """Generates random tokenized sequences for training."""

    def __init__(self, vocab_size, seq_len, num_samples):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        y = torch.cat(
            [x[1:], torch.tensor([0])]
        )  # Shifted left by 1 (next token prediction)
        return x, y


def train(model, dataloader, optimizer, criterion, device, num_epochs=3):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)  # (Batch, Seq Len, Vocab Size)

            loss = criterion(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )  # Flatten for loss calc
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch+1} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}"
                )

        print(
            f"Epoch {epoch+1} Completed | Avg Loss: {total_loss / len(dataloader):.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train MiniTransformer")
    parser.add_argument(
        "--processor", required=True, help="Path to SentencePiece tokenizer model."
    )
    parser.add_argument(
        "--embed-dim", type=int, default=256, help="Embedding dimension size."
    )
    parser.add_argument(
        "--n-heads", type=int, default=8, help="Number of attention heads."
    )
    parser.add_argument(
        "--ff-dim", type=int, default=512, help="Feed-forward network dimension."
    )
    parser.add_argument(
        "--n-layers", type=int, default=4, help="Number of transformer layers."
    )
    parser.add_argument(
        "--n-seq-len", type=int, default=128, help="Maximum sequence length."
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument(
        "--num-epochs", type=int, default=3, help="Number of training epochs."
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument(
        "--num-samples", type=int, default=10000, help="Number of training samples."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    processor = SentencePieceProcessor(model_file=args.processor)
    vocab_size = processor.vocab_size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MiniTransformer(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.n_heads,
        ff_dim=args.ff_dim,
        num_layers=args.n_layers,
        max_seq_len=args.n_seq_len,
    ).to(device)

    dataset = DummyDataset(vocab_size, args.n_seq_len, args.num_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print("Starting training...")
    train(model, dataloader, optimizer, criterion, device, num_epochs=args.num_epochs)
    print("Training complete!")
