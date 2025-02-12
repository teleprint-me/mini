"""
Copyright © 2023 Austin Berrio
Script: mini.cli.train
Description: Simple pre-training loop for text-to-text generation.
"""

from sentencepiece import SentencePieceProcessor

from mini.common.args import TransformerArgs
from mini.config import (
    ConfigCriterion,
    ConfigOptimizer,
    ConfigRuntime,
    ConfigScheduler,
    ConfigTransformer,
)
from mini.data.loader import JsonDatasetLoader, TextDatasetLoader
from mini.engine.optimizer_manager import EngineOptimizerManager
from mini.engine.state import EngineState
from mini.engine.trainer import EngineTrainer

if __name__ == "__main__":
    # Parse arguments
    args = TransformerArgs("Mini Training Tool").parse_args("train")

    # Load runtime configuration
    runtime = ConfigRuntime(seed=args.seed)
    runtime.seed_all()

    # Load model tokenizer
    processor = SentencePieceProcessor(model_file=args.processor)

    # Dynamically load Dataset & DataLoader
    dataset = None
    if args.dataset.endswith(".json"):
        dataset = JsonDatasetLoader(
            file_path=args.dataset,
            processor=processor,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            schema_path=args.schema_path,
            verbose=args.verbose,
        )
    else:  # Assume plaintext file
        dataset = TextDatasetLoader(
            file_path=args.dataset,
            processor=processor,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            batch_stride=args.batch_stride,
            verbose=args.verbose,
        )

    # Load Transformer Config
    config = ConfigTransformer(
        pad_id=max(processor.pad_id(), 0),
        vocab_size=processor.vocab_size(),
        max_seq_len=args.max_seq_len,
        embed_dim=args.embed_dim,
        theta=args.rope_theta,
        num_blocks=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        eps=args.eps,
        dropout=args.dropout,
        bias=args.bias,
    )

    # Load optimizer config
    config_optimizer = ConfigOptimizer(
        type=args.optimizer,
        lr=args.lr,
        eps=args.eps,
        amsgrad=args.amsgrad,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        dampening=args.dampening,
        nesterov=args.nesterov,
    )

    # Load scheduler config
    config_scheduler = ConfigScheduler(
        type=args.scheduler,
        step_size=args.step_size,
        gamma=args.gamma,
        T_max=args.t_max,
        eta_min=args.eta_min,
        start_factor=args.start_factor,
        total_iters=args.total_iters,
    )

    # Load criterion config
    config_criterion = ConfigCriterion(
        type=args.criterion,
        ignore_index=max(processor.pad_id(), 0),
        reduction=args.reduction,
    )

    # Load optimization manager
    manager = EngineOptimizerManager(
        config_optimizer=config_optimizer,
        config_scheduler=config_scheduler,
        config_criterion=config_criterion,
        verbose=args.verbose,
    )

    # Load state manager
    state = EngineState(
        path=args.model,
        config=config,
        manager=manager,
        runtime=runtime,
        verbose=args.verbose,
    )

    # Load trainer
    trainer = EngineTrainer(
        processor=processor,
        dataset=dataset,
        state=state,
        verbose=args.verbose,
    )

    trainer.train(
        num_epochs=args.num_epochs,
        save_every=args.save_every,
        grad_accum_steps=args.grad_accum_steps,
    )
