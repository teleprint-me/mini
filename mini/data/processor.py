"""
Copyright © 2023 Austin Berrio
Module: mini.data.processor
Description: Handles text and JSON data processing for NLP tasks.
"""

import logging
from typing import Dict, List

import torch
from sentencepiece import SentencePieceProcessor

from mini.common.logger import get_logger

# Type Annotations
JsonDataset = List[Dict[str, str]]
EncodedDataset = List[Dict[str, List[int]]]


class DatasetProcessor:
    """Base processor providing utility functions for text processing."""

    def __init__(
        self,
        processor: SentencePieceProcessor,
        max_seq_len: int,
        verbose: bool = False,
    ):
        self.processor = processor
        self.max_seq_len = max_seq_len
        self.pad_id = max(0, processor.pad_id())

        level = logging.DEBUG if verbose else logging.INFO
        self.logger = get_logger(self.__class__.__name__, level=level)

    def pad(self, sequence: List[int]) -> List[int]:
        """Pads a sequence to max_seq_len with a given pad_token."""
        return sequence + [self.pad_id] * (self.max_seq_len - len(sequence))

    def batch(
        self,
        sequences: List[Dict[str, List[int]]],
        batch_size: int = 8,
    ) -> List[Dict[str, torch.Tensor]]:
        """Batches tokenized data into PyTorch tensors."""

        batches = []

        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]

            _size = (len(batch), self.max_seq_len)
            _input = torch.full(_size, self.pad_id, dtype=torch.long)
            _target = torch.full(_size, self.pad_id, dtype=torch.long)

            for j, pair in enumerate(batch):
                _input[j] = torch.tensor(pair["input"], dtype=torch.long)
                _target[j] = torch.tensor(pair["target"], dtype=torch.long)

            batches.append({"input": _input, "target": _target})

        return batches


class TextDatasetProcessor(DatasetProcessor):
    """Processor for tokenizing and batching free-form text data."""

    def unsupervised(self, sequence: List[int]) -> List[Dict[str, List[int]]]:
        """Generates training pairs for unsupervised next-token prediction."""

        pairs = []

        for i in range(1, len(sequence)):
            _input = self.pad(sequence[:i])
            _target = self.pad(sequence[i : i + 1])

            assert len(_input) == len(_target), "Sequences must be the same shape"

            pairs.append({"input": _input, "target": _target})

        return pairs

    def supervised(self, sequence: List[int]) -> List[Dict[str, List[int]]]:
        """Generates training pairs for supervised next-token prediction."""

        pairs = []

        for i in range(1, len(sequence)):
            input_seq = self.pad(sequence[:i])
            target_seq = self.pad(sequence[: i + 1])
            assert len(input_seq) == len(target_seq), "Sequences must be the same shape"
            pairs.append({"input": input_seq, "target": target_seq})

        return pairs

    def next_token(
        self, sequence: List[int], supervised: bool = False
    ) -> List[Dict[str, List[int]]]:
        """Wrapper function that calls the appropriate sequence generation function."""
        # NOTE: There's a bug where tokens at the tail end may be unintentionally clipped.
        # This usually happens when the max seq len is less than the input len and
        # occassionally when max seq len is not evenly divisible by the input len.
        if supervised:
            return self.supervised(sequence)
        else:
            return self.unsupervised(sequence)

    def encode(
        self,
        raw_text: str,
        supervised: bool = False,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[Dict[str, List[int]]]:
        """Generates progressive input-target pairs from text."""

        tokens = self.processor.encode(raw_text, add_bos=add_bos, add_eos=add_eos)

        # Text is short, use recursive unmasking
        if len(tokens) <= self.max_seq_len:
            return self.next_token(tokens, supervised)

        # Text is long, chunk it into max_seq_len-sized pieces
        sequences = []
        for i in range(0, len(tokens), self.max_seq_len):
            window = tokens[i : i + self.max_seq_len]
            sequences = self.next_token(window, supervised)
            sequences.extend(sequences)

        return sequences


class JsonDatasetProcessor(DatasetProcessor):
    """Processor for tokenizing structured instruction-response JSON datasets."""

    def tokenize(
        self,
        json_dataset: JsonDataset,
        max_seq_len: int,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> EncodedDataset:
        """Tokenizes structured instruction-response pairs."""
        pad_id = self.processor.pad_id()
        if pad_id < 0:
            pad_id = 0  # Ensure a valid pad token

        sequences = []
        for entry in json_dataset:
            instruction = entry.get("instruction", "").strip()
            response = entry.get("response", "").strip()

            if not instruction or not response:
                self.logger.warning(f"Skipping entry with missing data: {entry}")
                continue

            input_tokens = self.processor.encode(
                instruction, add_bos=add_bos, add_eos=False
            )
            target_tokens = self.processor.encode(
                response, add_bos=False, add_eos=add_eos
            )

            sequences.append(
                {
                    "input": self.pad_or_truncate(input_tokens, max_seq_len, pad_id),
                    "target": self.pad_or_truncate(target_tokens, max_seq_len, pad_id),
                }
            )
        return sequences
