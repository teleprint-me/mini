"""
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

    def forward(self, x):
        B, T, C = x.shape  # Batch, Sequence Length, Embedding Dimension
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
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

    def forward(self, x):
        x = self.norm1(x + self.attn(x))  # Residual connection
        x = self.norm2(x + self.ff(x))  # Residual connection
        return x


class MiniTransformer(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_seq_len
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)  # Final projection

    def forward(self, x):
        B, T = x.shape  # Batch, Sequence Length
        x = self.embed(x) + self.pos_embed[:, :T, :]
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x)  # Output logits


def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--processor", required=True, help="Path to sentencepiece tokenizer model."
    )
    parser.add_argument("--model", required=True, help="Path to trained model.")
    parser.add_argument(
        "--embed-dim", default=256, help="Size of embeddings dimensions."
    )
    parser.add_argument("--n-heads", default=8, help="Number of heads.")
    parser.add_argument(
        "--ff-dim", default=512, help="Size of feed-forward dimensions."
    )
    parser.add_argument("--n-layers", default=4, help="Number of layers.")
    parser.add_argument("--n-seq-len", default=128, help="Max sequence length.")
    return parser.parse_args()


# Example Usage
if __name__ == "__main__":
    args = parse_args()

    processor = SentencePieceProcessor(args.processor)
    vocab_size = processor.vocab_size()  # The vocab size for this model is 32000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniTransformer(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.n_heads,
        ff_dim=args.ff_dim,
        num_layers=args.n_layers,
        max_seq_len=args.n_seq_len,
    ).to(device)

    x = torch.randint(0, vocab_size, (2, args.n_seq_len)).to(
        device
    )  # Simulated batch of tokenized input
    logits = model(x)  # (Batch, Seq Len, Vocab Size)
    print(logits.shape)  # Expected output: (2, 128, 32000)
