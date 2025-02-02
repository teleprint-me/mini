"""
Copyright © 2023 Austin Berrio
Script: mini.transformer.infer
Description: Simple completion for text-to-text generation with streaming output.
"""

import argparse
import os
import sys

import torch
from sentencepiece import SentencePieceProcessor

from mini.common.args import TransformerArgs
from mini.transformer.model import MiniTransformer, TransformerConfig
from mini.transformer.train import load_checkpoint


def generate(
    model: MiniTransformer,
    processor: SentencePieceProcessor,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 10,
    device: torch.device = "cpu",
) -> str:
    model.eval()  # Set model to evaluation mode

    # Encode prompt
    input_ids = processor.encode(prompt, add_bos=True, add_eos=False)
    print("Encoded input IDs:", input_ids)  # Debug
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    print("Initial input tensor shape:", input_tensor.shape)  # Debug

    generated_tokens = input_ids[:]  # Copy input tokens

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(input_tensor)  # Forward pass
            logits = logits[:, -1, :] / temperature  # Apply temperature scaling
            probs = torch.softmax(logits, dim=-1)

            # Top-k sampling
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            next_token = top_k_indices[
                0, torch.multinomial(top_k_probs, 1).item()
            ].item()

            output_text = processor.decode(next_token)
            print(output_text, end="")  # Stream output
            sys.stdout.flush()

            if next_token == processor.eos_id():
                print("\nEOS token encountered, stopping generation.")
                break  # Stop if EOS token is generated

            generated_tokens.append(next_token)

            # Update input tensor for next step
            input_tensor = torch.tensor(
                [generated_tokens], dtype=torch.long, device=device
            )

    print("\nEncoded output IDs:", generated_tokens)  # Debug
    return processor.decode(generated_tokens)


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on MiniTransformer.")
    parser.add_argument("--model", required=True, help="Path to trained model.")
    parser.add_argument(
        "--processor", required=True, help="Path to SentencePiece processor model."
    )
    parser.add_argument(
        "--prompt", required=True, help="Input text prompt for generation."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature."
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-k sampling size.")
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
        default=32,
        help="Head dimension size (Default: 32).",
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
        help="Theta for RoPE positional encoding (Default: 10000.0).",
    )
    parser.add_argument(
        "--bias", action="store_true", help="Use bias in FFN (Default: False)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load processor
    processor = SentencePieceProcessor(model_file=args.processor)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    mini = MiniTransformer(config).to(device)
    model, _ = load_checkpoint(args.model, mini, None)

    # Run inference
    output = generate(
        model,
        processor,
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    print("\nGenerated Output:", output)
