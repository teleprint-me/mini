"""
Copyright © 2023 Austin Berrio
Script: mini.cli.infer
Description: Simple completion for text-to-text generation with streaming output.
"""

import sys

from sentencepiece import SentencePieceProcessor

from mini.common.args import TransformerArgs
from mini.transformer.generator import (
    DEFAULT_PRETOKENIZER,
    GeneratorConfig,
    MiniGenerator,
)
from mini.transformer.manager import MiniManager
from mini.transformer.model import MiniConfig, MiniRuntime
from mini.transformer.sampler import MiniSampler, SamplerConfig
from mini.transformer.state import MiniState

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

    # Load sampler config
    sampler_config = SamplerConfig(
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        greedy=args.greedy,
        pad_id=max(processor.pad_id(), 0),
        verbose=args.verbose,
    )

    # Load sampler state
    sampler = MiniSampler(config=sampler_config)

    # Load generator config
    generator_config = GeneratorConfig(
        state=state,
        sampler=sampler,
        runtime=runtime,
        processor=processor,
        pre_tokenizer=DEFAULT_PRETOKENIZER,
    )

    # Load generator
    generator = MiniGenerator(config=generator_config)

    # Run inference
    print(f"\033[32;1;1m{args.prompt}\033[0m ", end="", flush=True)  # Output prompt
    for token in generator.stream(args.prompt, max_tokens=args.max_tokens):
        sys.stdout.write(token)
        sys.stdout.flush()
    print()  # Pad output
