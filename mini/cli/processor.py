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

    sp = SentencePieceProcessor(model_file=args.model)
    tp = TextDatasetProcessor(
        sp,
        args.max_seq_len,
        args.batch_size,
        args.add_bos,
        args.add_eos,
        args.supervise,
        args.verbose,
    )
    # Read input from a file or use the raw text input
    text = open_file(args.input) or args.input

    if args.verbose:
        _input = sp.encode(
            text,
            out_type=int,
            add_bos=args.add_bos,
            add_eos=args.add_eos,
        )
        _tokens = sp.encode(
            text,
            out_type=str,
            add_bos=args.add_bos,
            add_eos=args.add_eos,
        )
        print(
            f"Input: [{', '.join(f'\033[32;1;1m{repr(token)}\033[0m' for token in _input)}]"
        )
        print(
            f"Tokens: [{', '.join(f'\033[33;1;1m{repr(token)}\033[0m' for token in _tokens)}]"
        )
        print()

    sequences = tp.encode(text)
    batches = tp.batch(sequences)
    print(f"Generated \033[32;1;1m{len(sequences)}\033[0m sequences.")
    print(f"Generated \033[35;1;1m{len(batches)}\033[0m batches.")
    print()

    for i, batch in enumerate(batches):
        print(f"\033[35;1;1mbatch\033[0m={i + 1}", end=", ")
        print(f"\033[32;1;1minput_shape\033[0m={batch['input'].shape}", end=", ")
        print(f"\033[36;1;1mtarget_shape\033[0m={batch['target'].shape}")
        if args.verbose:
            print(f"\033[32;1;1minput\033[0m={batch['input']}")
            print(f"\033[36;1;1mtarget\033[0m={batch['target']}")
            print()
        assert len(batch["input"].shape) == len(batch["target"].shape), "Shape mismatch"


if __name__ == "__main__":
    main()
