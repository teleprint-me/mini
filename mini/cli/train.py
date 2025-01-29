"""
Copyright © 2023 Austin Berrio
Script: mini.cli.train
Description: Simple training loop for text-to-text generation.
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from sentencepiece import SentencePieceProcessor
from torch.utils.data import DataLoader, Dataset

from mini.common.json import JsonUtils
from mini.data.processor import EncodedDataset, MiniDataProcessor
from mini.model.transformer import MiniTransformer


class JsonDataset(Dataset):
    """Custom dataset class for loading tokenized instruction-response pairs."""

    def __init__(self, encoded_dataset: EncodedDataset):
        self.encoded_dataset = encoded_dataset

    def __len__(self):
        return len(self.encoded_dataset)

    def __getitem__(self, idx: int):
        item = self.encoded_dataset[idx]
        return torch.tensor(item["input"], dtype=torch.long), torch.tensor(
            item["target"], dtype=torch.long
        )


# NOTE: The optimizers base class is _Loss, but this is easier since it inherits from Module anyways.
def train(
    model_path: str,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.LRScheduler,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int = 10,
    save_every: int = 10,
):
    # Load the model if a checkpoint exists
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, weights_only=True))

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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"[Epoch {epoch+1}/{num_epochs}] Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}"
                )

        scheduler.step()
        print(
            f"Epoch {epoch+1} Completed | Avg Loss: {total_loss / len(dataloader):.4f}"
        )
        # Save the model periodically
        if (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train MiniTransformer")
    parser.add_argument(
        "--processor", required=True, help="Path to SentencePiece tokenizer model."
    )
    parser.add_argument(
        "--schema", required=True, help="Path to the schema file for the dataset."
    )
    parser.add_argument(
        "--dataset", required=True, help="Path to the filtered dataset JSON."
    )
    parser.add_argument(
        "--model", required=True, help="Path to save or load the model."
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
        "--num-epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--save-every", type=int, default=10, help="Save model every N epochs."
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load tokenizer
    processor = SentencePieceProcessor(model_file=args.processor)

    # Load dataset
    json_utils = JsonUtils()  # Handles errors, validation, etc. internally
    raw_dataset = json_utils.load_json(args.dataset)
    # Validate dataset
    raw_schema = json_utils.load_json(args.schema)
    json_utils.validate_json(raw_dataset, raw_schema)

    # Process dataset
    data_processor = MiniDataProcessor(processor)
    encoded_dataset = data_processor.tokenize(raw_dataset, max_length=args.n_seq_len)

    # Wrap into a PyTorch Dataset & DataLoader
    dataset = JsonDataset(encoded_dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model & Training Setup
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

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    criterion = nn.CrossEntropyLoss()

    print("Starting training...")
    train(
        model_path=args.model,
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=args.num_epochs,
        save_every=args.save_every,
    )
    print("Training complete!")
