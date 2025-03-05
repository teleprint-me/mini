"""
Copyright © 2023 Austin Berrio
Script: mini.cli.processor
Description: Script for experimenting with next-token and full-target supervision.
"""

import os

import torch
from sentencepiece import SentencePieceProcessor

from mini.args.processor import TextProcessorArgs


def open_file(path: str) -> str:
    """Reads text from a file or returns an empty string if the file is missing."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return ""


def pad_sequence(
    tokens: list[int], pad_token: int, max_seq_len: int, offset: int
) -> list[int]:
    """Pads a sequence to max_seq_len with a given pad_token."""
    assert max_seq_len > 0, "max_seq_len must be greater than 0"
    assert max_seq_len >= offset, "max_seq_len must be greater than or equal to offset"
    return tokens + [pad_token] * (max_seq_len - offset)


def generate_next_token_sequences(tokens, pad_token=0, max_seq_len=128):
    """
    Generates training pairs for next-token prediction.

    Args:
        tokens (list[int]): Full tokenized sequence.
        pad_token (int): Placeholder token for masked values.
        max_seq_len (int): Maximum sequence length.

    Returns:
        list[dict]: List of {"input": ..., "target": ...} dictionaries.
    """
    sequences = []
    length = len(tokens)

    for i in range(1, length):
        input_seq = pad_sequence(tokens[:i], pad_token, max_seq_len, i)
        target_seq = pad_sequence(tokens[i : i + 1], pad_token, max_seq_len, 1)
        assert len(input_seq) == len(target_seq), "Sequences must be the same shape"
        sequences.append({"input": input_seq, "target": target_seq})

    return sequences


def generate_next_token_supervision(tokens, pad_token=0, max_seq_len=128):
    """
    Generates training pairs where the target is the full sequence.

    Args:
        tokens (list[int]): Full tokenized sequence.
        pad_token (int): Placeholder token for masked values.
        max_seq_len (int): Maximum sequence length.

    Returns:
        list[dict]: List of {"input": ..., "target": ...} dictionaries.
    """
    sequences = []
    length = len(tokens)

    for i in range(1, length):
        input_seq = pad_sequence(tokens[:i], pad_token, max_seq_len, i)
        target_seq = pad_sequence(tokens[: i + 1], pad_token, max_seq_len, i + 1)
        assert len(input_seq) == len(target_seq), "Sequences must be the same shape"
        sequences.append({"input": input_seq, "target": target_seq})

    return sequences


def generate_progressive_sequences(
    tokens, pad_token=0, max_seq_len=128, supervised=False
):
    """
    Wrapper function that calls the appropriate sequence generation function.

    Args:
        tokens (list[int]): Full tokenized sequence.
        pad_token (int): Placeholder token for masked values.
        max_seq_len (int): Maximum sequence length.
        supervised (bool): Whether to use next-token prediction or full-sequence supervision.

    Returns:
        list[dict]: List of {"input": ..., "target": ...} dictionaries.
    """
    # NOTE: There's a bug where tokens at the tail end may be unintentionally clipped.
    # This usually happens when the max seq len is less than the input len and
    # occassionally when max seq len is not evenly divisible by the input len.
    if supervised:
        return generate_next_token_supervision(tokens, pad_token, max_seq_len)
    else:
        return generate_next_token_sequences(tokens, pad_token, max_seq_len)


def generate_training_data(tokens, pad_token=0, max_seq_len=128, supervised=False):
    """
    Generates progressive input-target pairs from text.

    Args:
        text (str): Raw input text.
        processor (SentencePieceProcessor): Tokenizer.
        max_seq_len (int): Maximum sequence length.
        pad_token (int): Token used for padding.

    Returns:
        list[dict]: List of {"input": ..., "target": ...} dictionaries.
    """
    # Case 1: Text is short, use recursive unmasking
    if len(tokens) <= max_seq_len:
        return [
            generate_progressive_sequences(tokens, pad_token, max_seq_len, supervised)
        ]

    # Case 2: Text is long, chunk it into max_seq_len-sized pieces
    batches = []
    for i in range(0, len(tokens), max_seq_len):
        window = tokens[i : i + max_seq_len]
        batch = generate_progressive_sequences(
            window, pad_token, max_seq_len, supervised
        )
        batches.append(batch)

    return batches


def main():
    args = TextProcessorArgs("Mini Processor CLI").parse_args()

    assert os.path.exists(args.model), "Model file does not exist!"
    assert args.input, "Input text or file must be provided!"
    assert args.max_seq_len % 2 == 0, "max_seq_len must be evenly divisible."

    processor = SentencePieceProcessor(model_file=args.model)

    # Read input from a file or use the raw text input
    text = open_file(args.input) or args.input

    out_type = str if args.out_type == "str" else int
    tokens = processor.encode(text, out_type=out_type)
    print(len(tokens), tokens)

    dataset = generate_training_data(
        tokens, pad_token=0, max_seq_len=args.max_seq_len, supervised=args.supervised
    )
    for i, block in enumerate(dataset):
        print(f"Block {i + 1} has {len(block) + 1} sequences")
        for j, sequence in enumerate(block):
            assert len(sequence["input"]) == len(sequence["target"]), "Shape mismatch"
            if args.verbose:
                print(f"input: {sequence['input']}")
                print(f"target: {sequence['target']}")


if __name__ == "__main__":
    main()
