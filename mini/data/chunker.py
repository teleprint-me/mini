"""
Copyright © 2023 Austin Berrio

Module: mini.data.chunker

Description: A class for processing and chunking file contents.
"""

import logging
import re
from typing import List

from sentencepiece import SentencePieceProcessor

from mini.common.logger import get_logger


class MiniDataChunker:
    """Class for processing and chunking file contents."""

    def __init__(
        self,
        processor: SentencePieceProcessor,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the FileChunker with the API instance and file path.

        Args:
            tokenizer: SentencePieceProcessor instance for tokenization.
            verbose (bool): Whether to enable verbose logging.
        """
        self.processor = processor
        self.verbose = verbose
        self.logger = get_logger(
            name=self.__class__.__name__,
            level=logging.DEBUG if verbose else logging.INFO,
        )

    def normalize_text(
        self,
        content: str,
        lowercase: bool = False,
        remove_punctuation: bool = False,
        preserve_structure: bool = True,
    ) -> str:
        """
        Normalize text by removing punctuation and converting to lowercase.
        Args:
            content (str): The text to be normalized.
            lowercase (bool): Whether to convert text to lowercase.
            remove_punctuation (bool): Whether to remove punctuation.
            preserve_structure (bool): Whether to preserve basic punctuation for structure.
        Returns:
            str: The normalized text.
        Raises:
            ValueError: If the content is not a string.
        """
        if not isinstance(content, str):
            raise ValueError("Content must be a string.")

        normalized = content.strip()
        if not normalized:
            self.logger.debug("Content is empty; skipping normalization.")
            return ""

        if lowercase:
            normalized = normalized.lower()

        if remove_punctuation:
            if preserve_structure:
                normalized = re.sub(r"[^\w\s.,?!'\"()]", "", normalized)
            else:
                normalized = re.sub(r"[^\w\s]", "", normalized)

        if self.verbose:
            self.logger.debug("Text normalization completed.")

        return normalized

    def chunk_text(
        self,
        content: str,
        batch_size: int = 256,
        chunk_size: int = 128,
        overlap: int = 0,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[str]:
        """
        Split text into chunks compatible with the model's embedding size and batch size.
        Args:
            content (str): The text to be chunked.
            batch_size (int): Physical batch size (defaults to 512).
            chunk_size (int): Maximum number of tokens per chunk (defaults to 256).
            overlap (int): Number of tokens to overlap between chunks (defaults to 0).
        Returns:
            List[str]: List of text chunks.
        """
        if not isinstance(content, str):
            raise ValueError("Content must be a string.")
        if not content:
            raise ValueError("Content cannot be empty.")

        # Determine the special token overhead
        if not (0 < chunk_size < batch_size):
            chunk_size = batch_size / 2
            self.logger.debug(
                f"Chunk size adjusted to {chunk_size} to fit within batch constraints."
            )
        if overlap >= chunk_size:
            overlap = chunk_size / 2
            self.logger.debug(
                f"Overlap size adjusted to {overlap} to fit within chunk constraints."
            )

        chunks = []
        tokens = self.processor.encode(content, add_bos=add_bos, add_eos=add_eos)
        self.logger.debug(f"Encoded tokens: {len(tokens)}")
        if not tokens:
            raise ValueError("Content is empty after encoding.")
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk_text = self.processor.decode(chunk_tokens)
            chunks.append(chunk_text)
            if self.verbose:
                self.logger.debug(
                    f"Chunk {i // (chunk_size - overlap)}: {len(chunk_tokens)} tokens."
                )

        if not chunks and content.strip():
            chunks.append(content.strip())
            self.logger.debug(
                "Content too short for chunking; returned as single chunk."
            )

        self.logger.debug(f"Generated {len(chunks)} chunks.")
        return chunks


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chunk a text into smaller pieces.")
    parser.add_argument(
        "--text-file", required=True, type=str, help="Path to the text file to chunk."
    )
    parser.add_argument(
        "--model-file", required=True, type=str, help="Path to the tokenizer model."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Physical batch size for processing (Default: 256; Set by llama-server).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="Size of each chunk in tokens (Default: 128; Must be less than batch size).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Overlap between chunks in tokens (Default: 0).",
    )
    parser.add_argument(
        "--add-bos",
        action="store_true",
        help="Add a beginning-of-sentence token to each chunk (Default: False).",
    )
    parser.add_argument(
        "--add-eos",
        action="store_true",
        help="Add an end-of-sentence token to each chunk (Default: False).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (Default: False).",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = get_logger(name="chunker", level=log_level)

    # Open the text file
    with open(args.text_file, "r", encoding="utf-8") as file:
        text = file.read()
    if args.verbose:
        logger.info(f"Text file content: {text[:100]}...")

    # Open the tokenizer model
    processor = SentencePieceProcessor(args.model_file)
    mini_chunker = MiniDataChunker(processor=processor, verbose=args.verbose)
    chunks = mini_chunker.chunk_text(
        text,
        args.batch_size,
        args.chunk_size,
        args.overlap,
        args.add_bos,
        args.add_eos,
    )
    for i, chunk in enumerate(chunks):
        logger.info(f"Chunk {i+1}: {chunk}")
