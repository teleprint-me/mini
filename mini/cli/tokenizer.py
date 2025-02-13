"""
Copyright © 2023 Austin Berrio
Script: mini.cli.tokenizer
Description: This script provides utilities for encoding, decoding, and saving tokenized text data.
"""

import os
from argparse import ArgumentParser, Namespace

from sentencepiece import SentencePieceProcessor

from mini.data.loader import TextDatasetLoader


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
        "--seq-length",
        action="store_true",
        help="Output the sequence length of the tokenized text (Default: False).",
    )
    parser.add_argument(
        "--clip",
        type=int,
        default=0,
        help="Clip output size (Default: 0).",
    )
    parser.add_argument(
        "--loader",
        action="store_true",
        help="Use a custom dataset loader (Default: False).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=128,
        help="Maximum sequence length (Default: 128).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of batches to process at once (Default: 8).",
    )
    parser.add_argument(
        "--batch-stride",
        type=int,
        default=32,
        help="The sequence window step size for batching (Default: 32).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (Default: False).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    assert args.input != args.output, "Input and output files must be different!"
    assert os.path.exists(args.model), "Model file does not exist!"
    assert args.clip >= 0, "Clip must be 0 or greater."

    # NOTE: This introduces a from time-of-check to time-of-use race condition.
    # This is just a quick and isolated hack to enable testing and should not be used in production.
    input_is_file = os.path.isfile(args.input)
    if input_is_file:
        text = clean_text(open_text(args.input))
    else:
        text = args.input

    processor = SentencePieceProcessor(model_file=args.model)
    if args.vocab_size:
        print("Vocab size:", processor.vocab_size())
        exit(0)

    if args.loader and not input_is_file:
        raise ValueError("Loader can only be used with file input.")
    if args.loader and input_is_file:
        loader = TextDatasetLoader(
            file_path=args.input,
            processor=processor,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            batch_stride=args.batch_stride,
            verbose=args.verbose,
        )
        print(
            f"Generated {len(loader.encoded)} sequences with stride {loader.batch_stride}"
        )
        if args.verbose:
            print("Encoded sequences:")
            for seq in loader.encoded:
                # NOTE: Input and target sequence lengths are equal.
                print(f"Input: len={len(seq['input'])}, input={seq['input']}")
                print(f"Target: len={len(seq['target'])}, target={seq['target']}")
                print()

        exit(0)

    if args.encode:
        tokens = processor.encode(text, out_type=str)
    else:
        tokens = processor.encode(text, out_type=None)

    if args.seq_length:
        print("Sequence Length:", len(tokens))
        exit(0)

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
