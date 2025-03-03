"""
Copyright © 2023 Austin Berrio
Script: mini.cli.process
Description: 
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


def generate_progressive_sequences(tokens, pad_token=0):
    """
    Generates progressively unmasked training sequences.

    Args:
        tokens (list[int]): Full tokenized sequence.
        pad_token (int): Placeholder token for masked values.

    Returns:
        list[dict]: List of {"input": ..., "target": ...} dictionaries.
    """
    sequences = []
    length = len(tokens)

    for i in range(1, length):
        input_seq = tokens[:i] + [pad_token] * (length - i)  # Progressive input
        target_seq = tokens  # Full sequence remains the same

        sequences.append({"input": input_seq, "target": target_seq})

    # append the full input and target as the final sequence
    sequences.append({"input": tokens, "target": tokens})

    return sequences


def main():
    args = TokenizerArgs("Mini Tokenizer CLI").parse_args()

    assert os.path.exists(args.model), "Model file does not exist!"
    assert args.input, "Input text or file must be provided!"

    processor = SentencePieceProcessor(model_file=args.model)

    # Read input from a file or use the raw text input
    text = open_file(args.input) or args.input
    print("text:", text)

    tokens = processor.encode(text)
    sequences = generate_progressive_sequences(tokens, 0)
    for seq in sequences:
        print("input:", seq["input"])
        print("target:", seq["target"])


if __name__ == "__main__":
    main()
