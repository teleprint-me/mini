"""
Copyright © 2023 Austin Berrio
Script: mini.cli.train
Description: Simple pre-training loop for text-to-text generation.
"""

import logging

from sentencepiece import SentencePieceProcessor

from mini.common.args import TransformerArgs
from mini.common.logger import get_logger
from mini.config.optimizer_manager import (
    ConfigCriterion,
    ConfigOptimizer,
    ConfigScheduler,
)
from mini.config.transformer import ConfigTransformer
from mini.data.loader import JsonDatasetLoader, TextDatasetLoader
from mini.engine.optimizer_manager import EngineOptimizerManager
from mini.engine.state import EngineState
from mini.engine.trainer import EngineTrainer

if __name__ == "__main__":
    # Parse arguments
    args = TransformerArgs("Mini Training Tool").parse_args("train")

    # Initialize logging
    logger = get_logger(__name__, level=logging.DEBUG if args.verbose else logging.INFO)

    # Load model tokenizer
    processor = SentencePieceProcessor(model_file=args.processor)

    # Define shared pad_id
    pad_id = max(processor.pad_id(), 0)

    # Dynamically load Dataset & DataLoader
    dataset = None
    if args.dataset.endswith(".json"):
        dataset = JsonDatasetLoader(
            file_path=args.dataset,
            processor=processor,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            schema_path=args.schema,
            verbose=args.verbose,
        )
    else:
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
        seed=args.seed,
        architecture=args.architecture,
        pad_id=pad_id,
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

    # Load optimizer config
    config_optimizer = ConfigOptimizer(
        type=args.optimizer,
        recurse=args.recurse,
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
        ignore_index=pad_id,
        reduction=args.reduction,
    )

    # Log training configuration
    logger.info("User Training Configuration:")
    logger.info(f"Model Architecture: {args.architecture}")
    logger.info(f"Number of Layers: {args.num_layers}")
    logger.info(f"Embedding Dim: {args.embed_dim}")
    logger.info(f"Feed Forward Dim: {args.ff_dim}, Expansion: {args.ff_mult}")
    logger.info(f"Optimizer: {args.optimizer}, LR: {args.lr}")
    logger.info(f"Scheduler: {args.scheduler}, Step Size: {args.step_size}")
    logger.info(f"Criterion: {args.criterion}, Reduction: {args.reduction}")
    logger.info(f"Training for {args.num_epochs} epochs...")
    logger.info(f"Max sequence length: {args.max_seq_len}")

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
