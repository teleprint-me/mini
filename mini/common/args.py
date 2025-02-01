"""
Copyright © 2023 Austin Berrio
Module: mini.common.args
Description: Contains the Args class for parsing command-line arguments.
"""

import argparse


class TransformerArgs:
    def __init__(self, description: str = "Mini CLI Tool"):
        self.parser = argparse.ArgumentParser(description=description)

    def parse_args(self) -> argparse.Namespace:
        self.add_required()
        self.add_optional()
        self.add_config()
        self.add_params()
        return self.parser.parse_args()

    def add_required(self) -> None:
        """Required arguments for model execution."""
        self.parser.add_argument(
            "--processor", required=True, help="Path to SentencePiece tokenizer model."
        )
        self.parser.add_argument(
            "--model", required=True, help="Path to save or load the model."
        )
        self.parser.add_argument(
            "--dataset", required=True, help="Path to a plaintext or JSON file."
        )

    def add_optional(self) -> None:
        """Optional arguments with safe defaults."""
        self.parser.add_argument(
            "--schema", default=None, help="Path to a JSON schema file for the dataset."
        )
        self.parser.add_argument(
            "-v", "--verbose", action="store_true", help="Enable verbose mode"
        )

    def add_config(self) -> None:
        """Model architecture and configuration."""
        self.parser.add_argument(
            "--embed-dim",
            type=int,
            default=512,
            help="Embedding dimension size (Default: 512).",
        )
        self.parser.add_argument(
            "--num-heads",
            type=int,
            default=8,
            help="Number of attention heads (Default: 8).",
        )
        self.parser.add_argument(
            "--head-dim",
            type=int,
            default=32,
            help="Head dimension size (Default: 32).",
        )
        self.parser.add_argument(
            "--num-layers",
            type=int,
            default=8,
            help="Number of transformer layers (Default: 8).",
        )
        self.parser.add_argument(
            "--ff-dim",
            type=int,
            default=512,
            help="Feed-forward network dimension (Default: 512).",
        )
        self.parser.add_argument(
            "--max-seq-len",
            type=int,
            default=512,
            help="Maximum sequence length (Default: 512).",
        )
        self.parser.add_argument(
            "--rope-theta",
            type=float,
            default=10000.0,
            help="Theta for RoPE positional encoding (Default: 10000.0).",
        )

        # Mutually exclusive bias flags
        group = self.parser.add_mutually_exclusive_group()
        group.add_argument(
            "--bias", action="store_true", help="Use bias in FFN (Default: False)."
        )
        group.add_argument(
            "--no-bias", action="store_false", dest="bias", help="Disable bias in FFN."
        )
        self.parser.set_defaults(bias=False)

    def add_params(self) -> None:
        """Training hyperparameters."""
        self.parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for reproducibility (Default: 42).",
        )
        self.parser.add_argument(
            "--batch-size",
            type=int,
            default=8,
            help="Batch size for training (Default: 8).",
        )
        self.parser.add_argument(
            "--batch-stride",
            type=int,
            default=32,
            help="Stride for batching the dataset (Default: 32).",
        )
        self.parser.add_argument(
            "--num-epochs",
            type=int,
            default=10,
            help="Number of training epochs (Default: 10).",
        )
        self.parser.add_argument(
            "--save-every",
            type=int,
            default=10,
            help="Save model every N epochs (Default: 10).",
        )
        self.parser.add_argument(
            "--eps",
            type=float,
            default=1e-8,
            help="Epsilon value for numerical stability (Default: 1e-8).",
        )
        self.parser.add_argument(
            "--weight-decay",
            type=float,
            default=0.0001,
            help="Weight decay for regularization (Default: 0.0001).",
        )
        self.parser.add_argument(
            "--amsgrad",
            action="store_true",
            help="Use AMSGrad for optimizer (Default: False).",
        )
        self.parser.add_argument(
            "--step-size",
            type=int,
            default=10,
            help="Learning rate scheduler step size (Default: 10).",
        )
        self.parser.add_argument(
            "--gamma",
            type=float,
            default=0.8,
            help="Learning rate scheduler gamma (Default: 0.8).",
        )
        self.parser.add_argument(
            "--lr", type=float, default=5e-4, help="Learning rate (Default: 5e-4)."
        )
        self.parser.add_argument(
            "--grad-accum-steps",
            type=int,
            default=1,
            help="Gradient accumulation steps (Default: 1).",
        )
        self.parser.add_argument(
            "--mixed-precision",
            type=str,
            choices=["none", "fp16", "bf16"],
            default="none",
            help="Enable mixed precision training: 'none', 'fp16' or 'bf16' (Default: none).",
        )
