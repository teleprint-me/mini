"""
Copyright © 2023 Austin Berrio
Module: mini.model.transformer
Description: A simple transformer model for quick and easy training.
"""

from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentencepiece import SentencePieceProcessor


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5  # Scaled dot-product attention

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x, mask=None):
        B, T, C = x.shape  # Batch, Sequence Length, Embedding Dimension
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale

        # Apply mask (if provided)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attn = Attention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x, mask=None):
        x = self.norm1(x + self.attn(x, mask))  # Residual connection
        x = self.norm2(x + self.ff(x))  # Residual connection
        return x


class MiniTransformer(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_seq_len
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)  # Final projection
        self.max_seq_len = max_seq_len

    def forward(self, x, mask=None):
        B, T = x.shape  # Batch, Sequence Length

        # Ignore PAD (0), but keep BOS (1) and EOS (2)
        if mask is None:
            mask = (x != 0).unsqueeze(1).unsqueeze(2)

        x = self.embed(x) + self.pos_embed[:, :T, :]
        for block in self.blocks:
            x = block(x, mask)  # Pass mask to each block
        x = self.norm(x)
        return self.head(x)  # Output logits


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Train MiniTransformer")
    parser.add_argument(
        "--processor", required=True, help="Path to SentencePiece tokenizer model."
    )
    parser.add_argument("--model", required=False, help="Path to trained model.")
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
    return parser.parse_args()


# Example Usage
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

    x = torch.randint(0, vocab_size, (2, args.n_seq_len), device=device)
    logits = model(x)
    print(logits.shape)  # Expected output: (2, 128, vocab_size)
