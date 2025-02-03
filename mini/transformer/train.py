"""
Copyright © 2023 Austin Berrio
Script: mini.transformer.train
Description: Simple pre-training loop for text-to-text generation.
"""

import random

import torch
from sentencepiece import SentencePieceProcessor

from mini.common.args import TransformerArgs
from mini.data.set import MiniDataset, MiniJsonDataset, MiniTextDataset
from mini.transformer.checkpoint import MiniCheckpoint
from mini.transformer.model import MiniTransformer, TransformerConfig
from mini.transformer.optimizer import MiniOptimizer
from mini.transformer.trainer import MiniTrainer


def seed_all(seed: int) -> None:
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Automatically detects and returns the best available device."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == "__main__":
    # Parse arguments
    args = TransformerArgs("Mini Training Tool").parse_args("train")
    # Set random seed
    seed_all(args.seed)
    # Load device
    device = get_device()

    # Load the model tokenizer
    processor = SentencePieceProcessor(model_file=args.processor)
    # Get the vocab size
    vocab_size = processor.vocab_size()
    # Get the pad id
    pad_id = max(processor.pad_id(), 0)

    # Setup PyTorch Dataset & DataLoader
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

    # Initialize the model and move it to the specified device
    model = MiniTransformer(config=config).to(device=device)

    # Initialize the optimizer
    optimizer = MiniOptimizer.create_optimizer(
        model=model,
        optimizer=args.optimizer,
        lr=args.lr,
        eps=args.eps,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad,
        momentum=args.momentum,
        dampening=args.dampening,
        nesterov=args.nesterov,
    )

    # Initialize the scheduler
    scheduler = MiniOptimizer.create_scheduler(
        optimizer=optimizer,
        scheduler=args.scheduler,
        step_size=args.step_size,
        gamma=args.gamma,
        t_max=args.t_max,
        eta_min=args.eta_min,
        start_factor=args.start_factor,
        total_iters=args.total_iters,
    )

    # Initialize the criterion
    criterion = MiniOptimizer.create_criterion(args.criterion, pad_id=pad_id)

    # Initialize the checkpoint
    checkpoint = MiniCheckpoint(
        path=args.model, optimizer=optimizer, verbose=args.verbose
    )

    # Load trainer
    trainer = MiniTrainer(
        processor=processor,
        dataset=dataset,
        model=model,
        config=config,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        checkpoint=checkpoint,
        device=device,
        verbose=args.verbose,
    )

    trainer.train(
        num_epochs=args.num_epochs,
        save_every=args.save_every,
        grad_accum_steps=args.grad_accum_steps,
    )
