"""
Copyright © 2023 Austin Berrio
Script: mini.transformer.train
Description: Simple pre-training loop for text-to-text generation.
"""

from sentencepiece import SentencePieceProcessor

from mini.common.args import TransformerArgs
from mini.data.set import MiniJsonDataset, MiniTextDataset
from mini.transformer.manager import (
    CriterionConfig,
    MiniManager,
    OptimizerConfig,
    SchedulerConfig,
)
from mini.transformer.model import MiniConfig, MiniRuntime
from mini.transformer.state import MiniState
from mini.transformer.trainer import MiniTrainer

if __name__ == "__main__":
    # Parse arguments
    args = TransformerArgs("Mini Training Tool").parse_args("train")

    # Load runtime configuration
    runtime = MiniRuntime(seed=args.seed)
    runtime.seed_all()

    # Load model tokenizer
    processor = SentencePieceProcessor(model_file=args.processor)

    # Dynamically load Dataset & DataLoader
    dataset = None
    if args.dataset.endswith(".json"):
        dataset = MiniJsonDataset(
            file_path=args.dataset,
            processor=processor,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            schema_path=args.schema_path,
            verbose=args.verbose,
        )
    else:  # Assume plaintext file
        dataset = MiniTextDataset(
            file_path=args.dataset,
            processor=processor,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            batch_stride=args.batch_stride,
            verbose=args.verbose,
        )

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

    # Load optimizer config
    optimizer_config = OptimizerConfig(
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
    scheduler_config = SchedulerConfig(
        type=args.scheduler,
        step_size=args.step_size,
        gamma=args.gamma,
        t_max=args.t_max,
        eta_min=args.eta_min,
        start_factor=args.start_factor,
        total_iters=args.total_iters,
    )

    # Load criterion config
    criterion_config = CriterionConfig(
        type=args.criterion,
        ignore_index=max(processor.pad_id(), 0),
        reduction=args.reduction,
    )

    # Load optimization manager
    manager = MiniManager(
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        criterion=criterion_config,
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

    # Load trainer
    trainer = MiniTrainer(
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
