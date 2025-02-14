"""
Copyright © 2023 Austin Berrio
Script: mini.cli.tokenizer
Description: This script provides utilities for encoding, decoding, and saving tokenized text data.
"""

import os

from sentencepiece import SentencePieceProcessor

from mini.args.tokenizer import TokenizerArgs
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


def main():
    args = TokenizerArgs("Mini Tokenizer CLI").parse_args()

    assert os.path.exists(args.model), "Model file does not exist!"
    assert args.clip >= 0, "Clip must be 0 or greater."

    processor = SentencePieceProcessor(model_file=args.model)
    if args.vocab_size:
        print("Vocab size:", processor.vocab_size())
        exit(0)

    # Ensure input text or file is provided.
    assert args.input, "Input text or file must be provided!"
    # Ensure input and output files are different.
    assert args.input != args.output, "Input and output files must be different!"

    # NOTE: This introduces a from time-of-check to time-of-use race condition.
    # This is just a quick and isolated hack to enable testing and should not be used in production.
    input_is_file = os.path.isfile(args.input)
    if input_is_file:
        text = clean_text(open_text(args.input))
    else:
        text = args.input

    if args.encode:
        tokens = processor.encode(text, out_type=str)
    else:
        tokens = processor.encode(text, out_type=None)

    if args.seq_length:
        print("Sequence Length:", len(tokens))
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

        if args.verbose:
            print("\nEncoded Sequences (Verbose Mode):\n")
            clip = len(loader.encoded[0]["input"]) if args.clip == 0 else args.clip

            for batch_idx, token_seq in enumerate(loader.encoded, start=1):
                print(f"Batch {batch_idx}:")
                print(
                    f"  Input  | seq_len={len(token_seq['input'])} | {token_seq['input'][: clip]}"
                )
                print(
                    f"  Target | seq_len={len(token_seq['target'])} | {token_seq['target'][: clip]}\n"
                )

        # Summarize the results.
        print(f"Total Sequences Generated: {len(loader.encoded)}")
        print(f"Max Sequence Length: {args.max_seq_len}")
        print(f"Batch Stride Used: {args.batch_stride}")

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
