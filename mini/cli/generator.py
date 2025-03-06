"""
Copyright © 2023 Austin Berrio
Script: mini.cli.generator
Description: Simple completion for text-to-text generation with streaming output.
"""

import sys

from sentencepiece import SentencePieceProcessor

from mini.args.transformer import TransformerArgs
from mini.config.generator import DEFAULT_PRETOKENIZER, ConfigGenerator, ConfigSampler
from mini.config.transformer import ConfigTransformer
from mini.engine.generator import EngineGenerator
from mini.engine.optimizer_manager import EngineOptimizerManager
from mini.engine.sampler import EngineSampler
from mini.engine.state import EngineState

if __name__ == "__main__":
    args = TransformerArgs("Mini Inference Tool").parse_args("generator")

    # Load model tokenizer
    processor = SentencePieceProcessor(model_file=args.processor)

    # Load Transformer Config
    config = ConfigTransformer(
        seed=args.seed,
        architecture=args.architecture,
        pad_id=max(processor.pad_id(), 0),
        vocab_size=processor.vocab_size(),
        max_seq_len=args.max_seq_len,
        embed_dim=args.embed_dim,
        rope_theta=args.rope_theta,
        num_mlp_layers=args.num_mlp_layers,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        ff_mult=args.ff_mult,
        mask_type=args.mask_type,
        eps=args.eps,
        dropout=args.dropout,
        bias=args.bias,
    )
    config.set_seed()

    # Load optimization manager
    manager = EngineOptimizerManager(
        config_optimizer=None,
        config_scheduler=None,
        config_criterion=None,
        verbose=args.verbose,
    )

    # Load state manager
    state = EngineState(
        path=args.model,
        config=config,
        manager=manager,
        verbose=args.verbose,
    )

    # Load sampler config
    config_sampler = ConfigSampler(
        seed=args.seed,
        pad_id=max(processor.pad_id(), 0),
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        greedy=args.greedy,
        verbose=args.verbose,
    )
    # Load sampler state
    sampler = EngineSampler(config=config_sampler)

    # Load generator config
    config_generator = ConfigGenerator(
        seed=args.seed,
        state=state,
        sampler=sampler,
        processor=processor,
        pre_tokenizer=DEFAULT_PRETOKENIZER,
    )

    # Load generator
    generator = EngineGenerator(config=config_generator)

    # Run inference
    print(f"\033[32;1;1m{args.prompt}\033[0m", end="", flush=True)  # Output prompt
    for token in generator.stream(args.prompt, max_tokens=args.max_tokens):
        sys.stdout.write(token)
        sys.stdout.flush()
    print()  # Pad output
