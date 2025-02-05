"""
Copyright © 2023 Austin Berrio
Script: mini.transformer.infer
Description: Simple completion for text-to-text generation with streaming output.
"""

import torch
from sentencepiece import SentencePieceProcessor

from mini.common.args import TransformerArgs
from mini.transformer.manager import MiniManager
from mini.transformer.model import MiniConfig, MiniRuntime
from mini.transformer.sampler import MiniSampler, SamplerConfig
from mini.transformer.state import MiniState


def generate(
    prompt: str,
    processor: SentencePieceProcessor,
    state: MiniState,
    sampler: MiniSampler,
    runtime: MiniRuntime,
) -> str:
    """Generates text using the trained transformer model with efficient token sampling."""

    # Load model state
    state.load(train=False)  # Ignore optimizer, scheduler, and criterion
    state.model.eval()

    # Fetch constants once
    pad_id = max(processor.pad_id(), 0)
    eos_id = processor.eos_id()

    # Encode prompt
    input_ids = processor.encode(prompt, add_bos=True, add_eos=False)
    generated_tokens = input_ids[:]  # Copy input tokens

    # Convert to tensor once and move to device
    input_tensor = torch.tensor(
        [input_ids], dtype=torch.long, device=runtime.device_type
    )

    with torch.no_grad():
        for _ in range(state.model.max_seq_len - len(generated_tokens)):
            logits = state.model(input_tensor)[:, -1, :]  # Forward pass
            next_token = sampler.sample(logits, past_tokens=generated_tokens)

            if next_token == pad_id:
                continue  # Ignore PAD token

            generated_tokens.append(next_token)

            # Efficient input tensor update: append only the new token
            input_tensor = torch.cat(
                [
                    input_tensor,
                    torch.tensor([[next_token]], device=runtime.device_type),
                ],
                dim=1,
            )

            # Decode only the latest token and print
            output_text = processor.decode([next_token])
            print(output_text, end="", flush=True)

            if next_token == eos_id:
                print("\nEOS token encountered, stopping generation.")
                break  # Stop if EOS token is generated

    return processor.decode(generated_tokens)


if __name__ == "__main__":
    args = TransformerArgs("Mini Inference Tool").parse_args("infer")

    # Load runtime configuration
    runtime = MiniRuntime(seed=args.seed)
    runtime.seed_all()

    # Load model tokenizer
    processor = SentencePieceProcessor(model_file=args.processor)

    # Load Transformer Config
    config = MiniConfig(
        vocab_size=processor.vocab_size(),
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        max_seq_len=args.max_seq_len,
        pad_id=max(processor.pad_id(), 0),
        eps=args.eps,
        theta=args.rope_theta,
        bias=args.bias,
    )

    # Load optimization manager
    manager = MiniManager(
        optimizer=None,
        scheduler=None,
        criterion=None,
        verbose=args.verbose,
    )

    # Load state manager
    state = MiniState(
        path=args.model,
        config=config,
        manager=manager,
        runtime=runtime,
        verbose=args.verbose,
    )

    sampler_config = SamplerConfig(
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        greedy=args.greedy,
        pad_id=max(processor.pad_id(), 0),
        verbose=args.verbose,
    )

    sampler = MiniSampler(config=sampler_config)

    # Run inference
    output = generate(
        args.prompt,
        processor,
        state=state,
        sampler=sampler,
        runtime=runtime,
    )
    print("\nGenerated Output:", output)
