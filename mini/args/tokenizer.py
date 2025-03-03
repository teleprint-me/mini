"""
Copyright © 2023 Austin Berrio
Module: mini.args.tokenizer
Description: Tokenizer-related arguments and configurations.
"""

from argparse import Namespace

from mini.args.base import BaseArgs


class TokenizerArgs(BaseArgs):
    def __init__(self, description: str = "Mini CLI Tool"):
        super().__init__(description)

    def parse_args(self) -> Namespace:
        self.add_required_args()
        self.add_tokenizer_args()
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
            choices=["int", "str", "bytes"],
            default="int",
            help="Output type of the encoded tokens (Default: int).",
        )
        self.parser.add_argument(
            "--vocab-size",
            action="store_true",
            help="Output the vocab size (Default: False)",
        )
        self.parser.add_argument(
            "--seq-len",
            action="store_true",
            help="Output the sequence length of the tokenized text (Default: False).",
        )
        self.parser.add_argument(
            "--splitlines",
            action="store_true",
            help="Process multiline input line by line (Default: False).",
        )
        self.parser.add_argument(
            "--preprocess",
            action="store_true",
            help="Strips newlines from input line by line (Default: False).",
        )
