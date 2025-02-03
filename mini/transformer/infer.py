"""
Copyright © 2023 Austin Berrio
Script: mini.transformer.infer
Description: Simple completion for text-to-text generation with streaming output.
"""

from typing import List, Optional

import torch
import torch.nn.functional as F
from sentencepiece import SentencePieceProcessor

from mini.common.args import TransformerArgs
from mini.transformer.checkpoint import MiniCheckpoint
from mini.transformer.model import MiniTransformer


def sample(
    logits: torch.Tensor,
    past_tokens: List[int],
    top_k: int = 50,
    top_p: float = 0.9,
    temperature: float = 0.8,
    repetition_penalty: float = 1.2,
    pad_id: Optional[int] = None,
) -> torch.Tensor:
    """Applies temperature, top-k, top-p filtering, and repetition penalty."""

    logits = logits / temperature  # Apply temperature scaling

    # Apply repetition penalty
    if past_tokens:
        for token in set(past_tokens):
            logits[:, token] /= repetition_penalty

    # Mask PAD tokens
    if pad_id is not None:
        logits[:, pad_id] = -float("inf")

    probs = F.softmax(logits, dim=-1)

    # Apply top-k filtering
    if top_k > 0:
        values, _ = torch.topk(probs, top_k)
        min_prob = values[:, -1].unsqueeze(-1)
        probs = torch.where(probs < min_prob, torch.zeros_like(probs), probs)

    # Apply top-p (nucleus) sampling
    if top_p > 0:
        sorted_probs, indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > top_p
        sorted_probs[mask] = 0
        sorted_probs = sorted_probs / sorted_probs.sum()
        probs.scatter_(-1, indices, sorted_probs)

    return torch.multinomial(probs, 1).item()  # Sample next token


def generate(
    model: MiniTransformer,
    processor: SentencePieceProcessor,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 10,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    device: torch.device = "cpu",
) -> str:
    """Generates text using the trained transformer model."""

    model.eval()
    pad_id = max(processor.pad_id(), 0)
    input_ids = processor.encode(prompt, add_bos=True, add_eos=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated_tokens = input_ids[:]  # Copy input tokens

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(input_tensor)[:, -1, :]  # Forward pass
            next_token = sample(
                logits,
                past_tokens=generated_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                pad_id=pad_id,  # Mask PAD token
            )

            # Decode and stream output
            if next_token == pad_id:
                continue  # Ignore PAD token

            output_text = processor.decode(next_token)
            print(output_text, end="", flush=True)

            if next_token == processor.eos_id():
                print("\nEOS token encountered, stopping generation.")
                break  # Stop if EOS token is generated

            generated_tokens.append(next_token)

            # Update input tensor for next step
            input_tensor = torch.tensor(
                [generated_tokens], dtype=torch.long, device=device
            )

    return processor.decode(generated_tokens)


if __name__ == "__main__":
    args = TransformerArgs("Mini Inference Tool").parse_args("infer")

    # Load processor
    processor = SentencePieceProcessor(model_file=args.processor)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint without requiring training parameters
    checkpoint = MiniCheckpoint(
        path=args.model,
        config=None,
        optimizer=None,
        device=device,
        verbose=args.verbose,
    )
    model, _ = checkpoint.load()

    # Run inference
    output = generate(
        model,
        processor,
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        device=device,
    )
    print("\nGenerated Output:", output)
