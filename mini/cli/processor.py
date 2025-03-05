"""
Copyright © 2023 Austin Berrio
Script: mini.cli.processor
Description: Script for experimenting with next-token and full-target supervision.
"""

import os

from sentencepiece import SentencePieceProcessor

from mini.args.processor import TextProcessorArgs
from mini.data.processor import TextDatasetProcessor


def open_file(path: str) -> str:
    """Reads text from a file or returns an empty string if the file is missing."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return ""


def main():
    args = TextProcessorArgs("Mini Processor CLI").parse_args()

    assert os.path.exists(args.model), "Model file does not exist!"
    assert args.input, "Input text or file must be provided!"
    assert args.max_seq_len % 2 == 0, "max_seq_len must be evenly divisible."

    sp = SentencePieceProcessor(model_file=args.model)
    tp = TextDatasetProcessor(sp, args.max_seq_len, args.verbose)
    # Read input from a file or use the raw text input
    text = open_file(args.input) or args.input

    encodings = sp.encode(text, out_type=int)
    tokens = sp.encode(text, out_type=str)
    print(f"Encodings: {encodings}")
    print(f"Tokens: {tokens}")

    sequences = tp.encode(text, args.supervised)
    print(f"Generated {len(sequences)} sequences...")
    batches = tp.batch(sequences, args.batch_size)
    print(f"Generated {len(batches)} batches...")
    for i, batch in enumerate(batches):
        print(
            f"batch={i + 1}, input={batch['input'].shape}, target={batch['target'].shape}"
        )
        if args.verbose:
            print(f"input={batch['input']}")
            print(f"target={batch['target']}")
        assert len(batch["input"].shape) == len(batch["target"].shape), "Shape mismatch"


if __name__ == "__main__":
    main()
