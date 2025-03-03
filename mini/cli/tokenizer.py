"""
Copyright © 2023 Austin Berrio
Script: mini.cli.tokenizer
Description: CLI for encoding, decoding, and saving tokenized text data.
"""

import json
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


def save_file(path: str, tokens: list, format: str = "json"):
    """Saves tokenized output in the specified format."""
    with open(path, "w", encoding="utf-8") as file:
        if format == "json":
            json.dump(tokens, file, indent=4)
        elif format == "newline":
            file.write("\n".join(map(str, tokens)))
        else:  # Default: space-separated tokens
            file.write(" ".join(map(str, tokens)))


def preprocess(text: str) -> str:
    """Joins lines into a single string with spaces to preserve structure."""
    return " ".join(text.splitlines())


def process(args, processor, text: str):
    """Processes text into tokenized form based on the specified output type."""
    if args.out_type == "raw":
        return text  # No processing, return raw text
    elif args.out_type == "str":
        return processor.encode(text, out_type=str)  # Returns token strings
    else:
        return processor.encode(text, out_type=None)  # Default: token IDs


def main():
    args = TokenizerArgs("Mini Tokenizer CLI").parse_args()

    assert os.path.exists(args.model), "Model file does not exist!"
    assert args.input, "Input text or file must be provided!"
    assert args.input != args.output, "Input and output files must be different!"
    assert not (
        args.preprocess and args.splitlines
    ), "preprocess and splitlines are mutually exclusive"

    processor = SentencePieceProcessor(model_file=args.model)

    if args.vocab_size:
        print(f"Vocab size: {processor.vocab_size()}")
        exit(0)

    # Read input from a file or use the raw text input
    text = open_file(args.input) or args.input

    if args.preprocess:
        text = preprocess(text)

    if args.seq_len:
        print(f"Sequence Length: {len(processor.encode(text))}")
        exit(0)

    if args.splitlines:
        results = [process(args, processor, line) for line in text.splitlines()]
        for line_tokens in results:
            print(line_tokens)
        if args.output:
            save_file(args.output, results, args.format)
        exit(0)

    tokens = process(args, processor, text)
    print(tokens)

    if args.output:
        save_file(args.output, tokens, args.format)


if __name__ == "__main__":
    main()
