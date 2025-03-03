"""
Copyright © 2023 Austin Berrio
Script: mini.cli.tokenizer
Description: This script provides utilities for encoding, decoding, and saving tokenized text data.
"""

import json
import os

from sentencepiece import SentencePieceProcessor

from mini.args.tokenizer import TokenizerArgs


def open_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return ""


def save_file(path: str, token_ids: list[int]):
    with open(path, "w", encoding="utf-8") as file:
        file.write(json.dumps(token_ids, indent=4))


def preprocess(text: str) -> str:
    body = ""
    for line in text.splitlines():
        body += " " + line if body else line
    return body


def process(args, processor, text: str) -> any:
    if args.out_type == "bytes":  # raw text
        return text
    elif args.out_type == "str":  # raw token
        return processor.encode(text, out_type=str)
    else:  # token id is int
        return processor.encode(text, out_type=None)


def main():
    args = TokenizerArgs("Mini Tokenizer CLI").parse_args()

    assert os.path.exists(args.model), "Model file does not exist!"
    assert args.input, "Input text or file must be provided!"
    assert args.input != args.output, "Input and output files must be different!"

    processor = SentencePieceProcessor(model_file=args.model)
    if args.vocab_size:
        print("Vocab size:", processor.vocab_size())
        exit(0)

    text = open_file(args.input)
    if not text:
        text = args.input

    if args.preprocess:
        text = preprocess(text)

    if args.seq_len:
        print(f"Sequence Length: {len(processor.encode(text))}")
        exit(0)

    if args.splitlines:
        for line in text.splitlines():
            print(process(args, processor, line))
        exit(0)

    tokens = process(args, processor, text)
    print(tokens)

    if args.output:
        save_file(args.output_file, tokens)


if __name__ == "__main__":
    main()
