"""
Copyright © 2023 Austin Berrio
Script: mini.cli.process
Description: Script for experimenting with next-token and full-target supervision.
"""

import os

from sentencepiece import SentencePieceProcessor

from mini.args.tokenizer import TokenizerArgs


def open_file(path: str) -> str:
    """Reads text from a file or returns an empty string if the file is missing."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return ""


def pad_sequence(tokens, pad_token, max_seq_len, offset):
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

    for i in range(1, max_seq_len):
        input_seq = pad_sequence(tokens[:i], pad_token, max_seq_len, i)
        target_seq = pad_sequence(
            [tokens[i] if i < length else pad_token], pad_token, max_seq_len, 1
        )
        sequences.append({"input": input_seq, "target": target_seq})

    return sequences


def generate_full_sequence_supervision(tokens, pad_token=0, max_seq_len=128):
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

    for i in range(1, max_seq_len):
        input_seq = pad_sequence(tokens[:i], pad_token, max_seq_len, i)
        target_seq = pad_sequence(tokens, pad_token, max_seq_len, length)
        sequences.append({"input": input_seq, "target": target_seq})

    return sequences


def generate_progressive_sequences(
    tokens, pad_token=0, max_seq_len=128, next_token_only=False
):
    """
    Wrapper function that calls the appropriate sequence generation function.

    Args:
        tokens (list[int]): Full tokenized sequence.
        pad_token (int): Placeholder token for masked values.
        max_seq_len (int): Maximum sequence length.
        next_token_only (bool): Whether to use next-token prediction or full-sequence supervision.

    Returns:
        list[dict]: List of {"input": ..., "target": ...} dictionaries.
    """
    if next_token_only:
        return generate_next_token_sequences(tokens, pad_token, max_seq_len)
    else:
        return generate_full_sequence_supervision(tokens, pad_token, max_seq_len)


def generate_training_data(processor, text, pad_token=0, max_seq_len=128):
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
    tokens = processor.encode(text)

    # Case 1: Text is short, use recursive unmasking
    if len(tokens) <= max_seq_len:
        return [generate_progressive_sequences(tokens, pad_token, max_seq_len)]

    # Case 2: Text is long, chunk it into max_seq_len-sized pieces
    batches = []
    for i in range(0, len(tokens), max_seq_len):
        window = tokens[i : i + max_seq_len]
        batch = generate_progressive_sequences(window, pad_token, max_seq_len)
        batches.append(batch)

    return batches


def main():
    args = TokenizerArgs("Mini Tokenizer CLI").parse_args()

    assert os.path.exists(args.model), "Model file does not exist!"
    assert args.input, "Input text or file must be provided!"

    processor = SentencePieceProcessor(model_file=args.model)

    # Read input from a file or use the raw text input
    text = open_file(args.input) or args.input

    dataset = generate_training_data(processor, text)
    for i, batch in enumerate(dataset):
        print(f"Batch {i + 1}")
        for j, sequence in enumerate(batch):
            assert len(sequence["input"]) == len(sequence["target"])
    print(f"Generated {len(dataset)} batches each with {len(dataset[0])} sequences.")


if __name__ == "__main__":
    main()
