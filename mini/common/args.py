"""
Copyright © 2023 Austin Berrio
Module: mini.common.args
Description: Contains the Args class for parsing command-line arguments.
"""

import argparse


class TransformerArgs:
    def __init__(self, description: str = "Mini CLI Tool"):
        self.parser = argparse.ArgumentParser(description=description)
        self.parser.add_argument(
            "-v", "--verbose", action="store_true", help="Enable verbose mode"
        )

    def parse_args(self, mode: str) -> argparse.Namespace:
        """
        Parses command-line arguments for a given mode ('train' or 'infer').

        Args:
            mode (str): The mode of execution ('train' or 'infer').

        Returns:
            argparse.Namespace: Parsed arguments.
        """
        self.add_common_args()

        if mode == "train":
            self.add_required_for_train()
            self.add_model_config()
            self.add_optimizer_params()
        elif mode == "infer":
            self.add_required_for_infer()
            self.add_model_config()
            self.add_sampling_args()
        else:
            raise ValueError("Invalid mode. Use 'train' or 'infer'.")

        return self.parser.parse_args()

    def add_common_args(self) -> None:
        """Common arguments shared across training and inference."""
        self.parser.add_argument(
            "--processor", required=True, help="Path to SentencePiece tokenizer model."
        )
        self.parser.add_argument(
            "--model", required=True, help="Path to save or load the model."
        )

    def add_required_for_train(self) -> None:
        """Required arguments for training mode."""
        self.parser.add_argument(
            "--dataset", required=True, help="Path to a plaintext or JSON dataset."
        )
        self.parser.add_argument(
            "--schema",
            default=None,
            help="Optional JSON schema for dataset validation.",
        )

    def add_required_for_infer(self) -> None:
        """Required arguments for inference mode."""
        self.parser.add_argument(
            "--prompt", required=True, help="Input text prompt for generation."
        )

    def add_sampling_args(self) -> None:
        """Arguments related to sampling during inference."""
        self.parser.add_argument(
            "--max-tokens",
            type=int,
            default=128,
            help="Maximum number of tokens to generate.",
        )
        self.parser.add_argument(
            "--temperature", type=float, default=0.8, help="Sampling temperature."
        )
        self.parser.add_argument(
            "--top-k", type=int, default=10, help="Top-k sampling size (Default: 10)."
        )
        self.parser.add_argument(
            "--top-p",
            type=float,
            default=0.9,
            help="Top-p sampling probability (Default: 0.9).",
        )
        self.parser.add_argument(
            "--repetition-penalty",
            type=float,
            default=1.0,
            help="Repetition penalty factor (Default: 1.0).",
        )

    def add_model_config(self) -> None:
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
            default=256,
            help="Feed-forward network dimension (Default: 256).",
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
            help="Theta for RoPE encoding (Default: 10000.0).",
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

    def add_optimizer_params(self) -> None:
        """Training hyperparameters."""
        self.parser.add_argument("--seed", type=int, default=42, help="Random seed.")
        self.parser.add_argument(
            "--batch-size", type=int, default=8, help="Batch size (Default: 8)."
        )
        self.parser.add_argument(
            "--batch-stride",
            type=int,
            default=32,
            help="Stride for dataset batching (Default: 32).",
        )
        self.parser.add_argument(
            "--num-epochs",
            type=int,
            default=50,
            help="Number of training epochs (Default: 50).",
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
            help="Epsilon for numerical stability (Default: 1e-8).",
        )
        self.parser.add_argument(
            "--weight-decay",
            type=float,
            default=1e-2,
            help="Weight decay (Default: 1e-2).",
        )
        self.parser.add_argument(
            "--amsgrad",
            action="store_true",
            help="Use AMSGrad optimizer (Default: False).",
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
            default=0.1,
            help="Learning rate scheduler gamma (Default: 0.1).",
        )
        self.parser.add_argument(
            "--lr", type=float, default=1e-4, help="Learning rate (Default: 1e-4)."
        )
        self.parser.add_argument(
            "--grad-accum-steps",
            type=int,
            default=1,
            help="Gradient accumulation steps (Default: 1).",
        )
