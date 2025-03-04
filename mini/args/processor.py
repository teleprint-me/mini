"""
Copyright © 2023 Austin Berrio
Module: mini.args.processor
Description: Data processor related arguments and configurations.
"""

from argparse import Namespace

from mini.args.base import BaseArgs


class TextProcessorArgs(BaseArgs):
    def __init__(self, description: str = "Mini CLI Tool"):
        super().__init__(description)

    def parse_args(self) -> Namespace:
        self.add_required_args()
        self.add_tokenizer_args()
        self.add_processor_args()
        return self.parser.parse_args()

    def add_required_args(self) -> None:
        self.parser.add_argument(
            "--model",
            required=True,
            help="Path to the SentencePiece model file.",
        )

    def add_tokenizer_args(self) -> None:
        self.parser.add_argument(
            "--input",
            type=str,
            help="Path to the input text or file.",
        )
        self.parser.add_argument("--output", help="Path to the tokenized output file.")
        self.parser.add_argument(
            "--out-type",
            choices=["int", "str"],
            default="bytes",
            help="Output type of the encoded tokens (Default: int).",
        )

    def add_processor_args(self) -> None:
        self.parser.add_argument(
            "--max-seq-len",
            type=int,
            default=10,
            help="Maximum sequence length (Default: 10).",
        )
        self.parser.add_argument(
            "--supervised",
            action="store_true",
            help="Enable next token supervision (Default: False).",
        )
