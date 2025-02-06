"""
Module: mini.data.tokenizer
Description: This module provides a tokenizer class for tokenizing text data.
"""

import os
from argparse import ArgumentParser, Namespace

from sentencepiece import SentencePieceProcessor


def open_text(file_path) -> str:
    print("Reading plaintext file for processing.")
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def save_tokens(tokens, output_file):
    print("Writing encoded plaintext file for observation.")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(" ".join(tokens))  # Save as space-separated tokens


def clean_text(text: str) -> str:
    # Keep only meaningful, non-empty lines
    return "\n".join([line.strip() for line in text.splitlines() if line.strip()])


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Tokenize text data using SentencePiece.")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the SentencePiece model file.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input text or file.",
    )
    parser.add_argument("--output", help="Path to the tokenized output file.")
    parser.add_argument(
        "--decode",
        action="store_true",
        help="Output decoded tokens (Default: False).",
    )
    parser.add_argument(
        "--encode",
        action="store_true",
        help="Set out_type to str when encoding (Default: False).",
    )
    parser.add_argument(
        "--vocab-size",
        action="store_true",
        help="Output the vocab size (Default: False)",
    )
    parser.add_argument(
        "--clip",
        type=int,
        default=0,
        help="Clip output size (Default: 0).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.input == args.output:
        raise ValueError("Do **not** output to input file!")

    if args.clip < 0:
        raise ValueError("Clip must be 0 or greater.")

    if os.path.exists(args.input):
        text = clean_text(open_text(args.input))
    else:
        text = args.input

    processor = SentencePieceProcessor(model_file=args.model)
    if args.vocab_size:
        print("Vocab size:", processor.vocab_size())

    if args.encode:
        tokens = processor.encode(text, out_type=str)
    else:
        tokens = processor.encode(text, out_type=None)

    if args.clip > 0:
        print("Encoded:", tokens[: args.clip])
    else:
        print("Encoded:", tokens)

    if args.decode:
        decoded = processor.decode(tokens)
        if args.clip > 0:
            print("Decoded:", decoded[: args.clip])
        else:
            print("Decoded:", decoded)

    if args.output:
        save_tokens(tokens, args.output_file)


if __name__ == "__main__":
    main()
