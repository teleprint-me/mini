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


def generate_progressive_sequences(tokens, pad_token=-1):
    """
    Generates progressively unmasked training sequences.

    Args:
        tokens (list[str]): Full tokenized sequence.
        pad_token (int): Placeholder token for masked values.

    Returns:
        list[tuple]: List of (input_sequence, target) pairs.
    """
    sequences = []
    length = len(tokens)

    for i in range(1, length):
        input_seq = tokens[:i] + [pad_token] * (length - i)
        target = tokens[i]  # Predict this token
        sequences.append({"input": input_seq, "target": target})

    return sequences


def main():
    args = TokenizerArgs("Mini Tokenizer CLI").parse_args()

    assert os.path.exists(args.model), "Model file does not exist!"
    assert args.input, "Input text or file must be provided!"

    processor = SentencePieceProcessor(model_file=args.model)

    # Read input from a file or use the raw text input
    text = open_file(args.input) or args.input
    tokens = processor.encode(text)

    sequences = generate_progressive_sequences(tokens, 0)
    print("sequences", len(sequences))
    print("input", sequences[0]["input"])
    print("target", sequences[0]["target"])
    decoded_in = processor.decode(sequences[0]["input"])
    decoded_tar = processor.decode(sequences[0]["target"])
    print("decoded input", decoded_in)
    print("decoded target", decoded_tar)
    # for seq in sequences:
    #     print(sequences)


if __name__ == "__main__":
    main()
