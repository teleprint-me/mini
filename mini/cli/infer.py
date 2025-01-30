"""
Script: mini.cli.infer
Description: Simple completion for text-to-text generation.
"""

import argparse
import os
import sys

import torch
from sentencepiece import SentencePieceProcessor

from mini.model.transformer import MiniTransformer


def generate(
    model: MiniTransformer,
    tokenizer: SentencePieceProcessor,
    prompt: str,
    max_tokens: int = 128,
    device: torch.device = "cpu",
) -> str:
    model.eval()  # Set model to evaluation mode

    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    print("Encoded input IDs:", input_ids)  # Debug
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    print("Initial input tensor shape:", input_tensor.shape)  # Debug

    generated_tokens = input_ids[:]  # Copy input tokens

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(input_tensor)  # Forward pass

            next_token = logits[:, -1, :].argmax(dim=-1).item()
            output_text = tokenizer.decode(next_token)
            print(output_text, end=" ")  # Debug
            sys.stdout.flush()

            if next_token == tokenizer.eos_id():
                print("EOS token encountered, stopping generation.")
                break  # Stop if EOS token is generated

            generated_tokens.append(next_token)

            # Update input tensor for next step
            input_tensor = torch.tensor(
                [generated_tokens], dtype=torch.long, device=device
            )

    output_text = tokenizer.decode(generated_tokens)
    print("Decoded output:", output_text)  # Debug
    return output_text


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on MiniTransformer.")
    parser.add_argument("--model", required=True, help="Path to trained model.")
    parser.add_argument(
        "--processor", required=True, help="Path to SentencePiece tokenizer model."
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


if __name__ == "__main__":
    args = parse_args()

    # Load tokenizer
    tokenizer = SentencePieceProcessor(model_file=args.processor)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MiniTransformer(
        vocab_size=tokenizer.vocab_size(),
        embed_dim=args.embed_dim,
        num_heads=args.n_heads,
        ff_dim=args.ff_dim,
        num_layers=args.n_layers,
        max_seq_len=args.n_seq_len,
    ).to(device)

    # Load the model and optimizer if a checkpoint exists
    if os.path.exists(args.model):
        print(f"Loading model from {args.model}")
        checkpoint = torch.load(args.model, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])

    # Run inference
    output = generate(
        model, tokenizer, args.prompt, max_tokens=args.max_tokens, device=device
    )
    print("Generated Output:", output)
