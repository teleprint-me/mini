"""
Module: mini.data.tokenizer
Description: This module provides a tokenizer class for tokenizing text data.
"""

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
        "--model-file", required=True, help="Path to the SentencePiece model file."
    )
    parser.add_argument(
        "--input-file", required=True, help="Path to the input text file."
    )
    parser.add_argument("--output-file", help="Path to the output tokenized file.")
    return parser.parse_args()


def main():
    args = parse_args()

    text = open_text(args.input_file)
    cleaned_text = clean_text(text)

    processor = SentencePieceProcessor(model_file=args.model_file)
    print("Vocab size:", processor.vocab_size())

    tokens = processor.encode(cleaned_text)
    print("Encoded:", tokens[:100])

    decoded = processor.decode(tokens)
    print("Decoded:", decoded[:100])

    if args.output_file:
        save_tokens(tokens, args.output_file)


if __name__ == "__main__":
    main()
