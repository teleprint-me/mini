"""
Copyright © 2023 Austin Berrio
Module: mini.args.transformer
Description: Contains the Args class for parsing command-line arguments.
"""

from argparse import Namespace

from mini.args.base import BaseArgs
from mini.models.factory import MODEL_REGISTRY


class TransformerArgs(BaseArgs):
    def __init__(self, description: str = "Mini CLI Tool"):
        super().__init__(description)

    def parse_args(self, mode: str) -> Namespace:
        """
        Parses command-line arguments for a given mode ('trainer' or 'generator').

        Args:
            mode (str): The mode of execution ('trainer' or 'generator').

        Returns:
            argparse.Namespace: Parsed arguments.
        """

        self.add_common_args()  # Shared tokenizer/model args

        if mode == "trainer":
            self.add_dataset_args()  # Dataset parameters
            self.add_model_config()  # Transformer architecture
            self.add_training_args()  # Training hyperparameters
            self.add_optimizer_args()  # Optimizer settings
            self.add_scheduler_args()  # Learning rate scheduling
            self.add_criterion_args()  # Loss function

        elif mode == "generator":
            self.add_inference_args()  # Input for inference
            self.add_model_config()  # Model architecture (shared with trainer)

        else:
            raise ValueError("Invalid mode. Use 'trainer' or 'generator'.")

        return self.parser.parse_args()

    def add_common_args(self) -> None:
        """Common arguments shared across training and inference."""

        self.parser.add_argument(
            "--processor", required=True, help="Path to SentencePiece tokenizer model."
        )
        self.parser.add_argument(
            "--model", required=True, help="Path to save or load the model."
        )
        self.parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    def add_dataset_args(self) -> None:
        """Arguments related to dataset loading and preprocessing."""

        self.parser.add_argument(
            "--dataset",
            required=True,
            help="Path to dataset file (.json, .txt).",
        )
        self.parser.add_argument(
            "--schema",
            default=None,
            help="Path to JSON schema for validation (only required for JSON datasets).",
        )
        self.parser.add_argument(
            "--batch-size",
            type=int,
            default=8,
            help="Batch size for training (Default: 8).",
        )
        self.parser.add_argument(
            "--add-bos",
            action="store_true",
            help="Include beginning-of-sequence token.",
        )
        self.parser.add_argument(
            "--add-eos",
            action="store_true",
            help="Include end-of-sequence token.",
        )
        self.parser.add_argument(
            "--supervise",
            action="store_true",
            help="Enable supervised training mode (targets include full input context).",
        )

    def add_inference_args(self) -> None:
        """Arguments required for inference (text generation)."""

        self.parser.add_argument(
            "--prompt",
            required=True,
            help="Input text prompt for generation.",
        )
        self.parser.add_argument(
            "--max-tokens",
            type=int,
            default=128,
            help="Maximum number of tokens to generate (Default: 128).",
        )
        self.parser.add_argument(
            "--temperature",
            type=float,
            default=0.8,
            help="Sampling temperature (Default: 0.8). Lower is more deterministic.",
        )
        self.parser.add_argument(
            "--top-k",
            type=int,
            default=50,
            help="Top-K sampling size (Default: 50). Lower values make generation more focused.",
        )
        self.parser.add_argument(
            "--top-p",
            type=float,
            default=0.9,
            help="Top-P (nucleus) sampling probability (Default: 0.9). Controls diversity.",
        )
        self.parser.add_argument(
            "--repetition-penalty",
            type=float,
            default=1.2,
            help="Penalty for repeated tokens (Default: 1.2). Higher values discourage repetition.",
        )
        self.parser.add_argument(
            "--greedy",
            action="store_true",
            help="Enable greedy decoding instead of sampling (Default: False).",
        )

    def add_model_config(self) -> None:
        """Model architecture and configuration."""

        self.parser.add_argument(
            "--architecture",
            type=str,
            choices=list(MODEL_REGISTRY.keys()),
            default="misty",
            help="Model architecture (Default: misty).",
        )

        # Encoder and Embedding
        self.parser.add_argument(
            "--max-seq-len",
            type=int,
            default=256,
            help="Maximum sequence length (Default: 256).",
        )
        self.parser.add_argument(
            "--embed-dim",
            type=int,
            default=512,
            help="Embedding dimension size (Default: 512).",
        )
        self.parser.add_argument(
            "--rope-theta",
            type=float,
            default=10000.0,
            help="Scale factor for RoPE encoding (Default: 10000.0).",
        )

        # Transformer Blocks
        self.parser.add_argument(
            "--num-mlp-layers",
            type=int,
            default=3,
            help="Number of MLP layers (Default: 3).",
        )
        self.parser.add_argument(
            "--num-layers",
            type=int,
            default=8,
            help="Number of transformer layers (Default: 8).",
        )
        self.parser.add_argument(
            "--num-heads",
            type=int,
            default=4,
            help="Number of attention heads (Default: 4).",
        )
        self.parser.add_argument(
            "--ff-dim",
            type=int,
            default=128,
            help="Feed-forward network dimension (Default: 128).",
        )
        self.parser.add_argument(
            "--ff-mult",
            type=float,
            default=4.0,
            help="Feed-forward network expansion factor (Default: 4.0).",
        )
        self.parser.add_argument(
            "--mask-type",
            type=str,
            default="causal",
            choices=["causal", "bidirectional"],
            help="Mask type for transformer (Default: causal).",
        )

        # Shared Layer Normalization
        self.parser.add_argument(
            "--eps",
            type=float,
            default=1e-8,
            help="Epsilon for numerical stability (Default: 1e-8).",
        )
        self.parser.add_argument(
            "--dropout",
            type=float,
            default=0.1,
            help="Dropout rate (Default: 0.1).",
        )
        self.parser.add_argument(
            "--bias", action="store_true", help="Use bias in FFN (Default: False)."
        )

    def add_training_args(self) -> None:
        """Training hyperparameters."""

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
            "--grad-accum-steps",
            type=int,
            default=1,
            help="Gradient accumulation steps (Default: 1).",
        )

    def add_optimizer_args(self) -> None:
        """Optimizer settings for training."""

        self.parser.add_argument(
            "--optimizer",
            choices=["adam", "adamw", "sgd"],
            default="adamw",
            help="Optimizer to use (Default: adamw).",
        )

        # General optimizer settings
        self.parser.add_argument(
            "--lr",
            type=float,
            default=1e-4,
            help="Optimizer learning rate (Default: 1e-4).",
        )
        self.parser.add_argument(
            "--weight-decay",
            type=float,
            default=0.0,
            help="Weight decay (Default: 0.0).",
        )
        self.parser.add_argument(
            "--recurse",
            action="store_false",
            help="Optimizer will yield model parameters recursively (Default: True).",
        )

        # Adam / AdamW specific params
        self.parser.add_argument(
            "--amsgrad",
            action="store_true",
            help="Use AMSGrad variant of Adam (Default: False).",
        )

        # SGD specific params
        self.parser.add_argument(
            "--momentum",
            type=float,
            default=0.0,
            help="Momentum factor for SGD (Default: 0.0).",
        )
        self.parser.add_argument(
            "--dampening",
            type=float,
            default=0.0,
            help="Dampening factor for SGD (Default: 0.0).",
        )
        self.parser.add_argument(
            "--nesterov",
            action="store_true",
            help="Enable Nesterov momentum for SGD (Default: False).",
        )

    def add_scheduler_args(self) -> None:
        """Learning rate scheduler configuration."""

        self.parser.add_argument(
            "--scheduler",
            choices=["none", "step", "cosine", "linear"],
            default="step",
            help="Learning rate scheduler type (Default: step).",
        )

        # StepLR scheduler
        self.parser.add_argument(
            "--step-size",
            type=int,
            default=10,
            help="StepLR: Learning rate step size (Default: 10).",
        )
        self.parser.add_argument(
            "--gamma",
            type=float,
            default=0.1,
            help="StepLR: Learning rate decay rate (Default: 0.1).",
        )

        # Cosine Annealing scheduler
        self.parser.add_argument(
            "--T-max",
            type=int,
            default=50,
            help="Cosine: Maximum number of iterations (Default: 50).",
        )
        self.parser.add_argument(
            "--eta-min",
            type=float,
            default=1e-6,
            help="Cosine: Minimum learning rate (Default: 1e-6).",
            dest="eta_min",
        )

        # TODO: Implement warmup support
        self.parser.add_argument(
            "--start-factor",
            type=float,
            default=0.1,
            help="Linear warmup start factor (Default: 0.1).",
        )
        self.parser.add_argument(
            "--total-iters",
            type=float,
            default=50,
            help="Linear warmup total iterations (Default: 50).",
        )

    def add_criterion_args(self) -> None:
        """Loss function selection."""

        self.parser.add_argument(
            "--criterion",
            choices=["cross_entropy", "mse", "mae"],
            default="cross_entropy",
            help="Loss function for training (Default: cross_entropy).",
        )
        self.parser.add_argument(
            "--reduction",
            choices=["mean", "sum", "none"],
            default="mean",
            help="Reduction method for loss calculation (Default: mean).",
        )
