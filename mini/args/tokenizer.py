"""
Copyright © 2023 Austin Berrio
Module: mini.args.tokenizer
Description: Tokenizer-related arguments and configurations.
"""

from argparse import ArgumentParser, Namespace


class TokenizerArgs:
    def __init__(self, description: str = "Mini CLI Tool"):
        self.parser = ArgumentParser(description=description)
        self.parser.add_argument(
            "-v", "--verbose", action="store_true", help="Enable verbose mode"
        )

    def parse_args(self) -> Namespace:
        self.add_required_args()
        self.add_tokenizer_args()
        self.add_loader_args()
        return self.parser.parse_args()

    def add_required_args(self) -> None:
        self.parser.add_argument(
            "--model",
            required=True,
            help="Path to the SentencePiece model file.",
        )
        self.parser.add_argument(
            "--input",
            type=str,
            help="Path to the input text or file.",
        )
        self.parser.add_argument("--output", help="Path to the tokenized output file.")

    def add_tokenizer_args(self) -> None:
        self.parser.add_argument(
            "--decode",
            type=str,
            default=None,
            help="Output decoded tokens (Default: None).",
        )
        self.parser.add_argument(
            "--out-type",
            choices=["int", "str"],
            default="int",
            help="Output type of the encoded tokens (Default: int).",
        )
        self.parser.add_argument(
            "--vocab-size",
            action="store_true",
            help="Output the vocab size (Default: False)",
        )
        self.parser.add_argument(
            "--seq-length",
            action="store_true",
            help="Output the sequence length of the tokenized text (Default: False).",
        )
        self.parser.add_argument(
            "--clip",
            type=int,
            default=0,
            help="Clip output size (Default: 0).",
        )

    def add_loader_args(self) -> None:
        self.parser.add_argument(
            "--loader",
            action="store_true",
            help="Use a custom dataset loader (Default: False).",
        )
        self.parser.add_argument(
            "--max-seq-len",
            type=int,
            default=128,
            help="Maximum sequence length (Default: 128).",
        )
        self.parser.add_argument(
            "--batch-size",
            type=int,
            default=8,
            help="Number of batches to process at once (Default: 8).",
        )
        self.parser.add_argument(
            "--batch-stride",
            type=int,
            default=32,
            help="The sequence window step size for batching (Default: 32).",
        )
