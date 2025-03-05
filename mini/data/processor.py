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

        log_level = logging.DEBUG if verbose else logging.INFO
        self.logger = get_logger(self.__class__.__name__, log_level)

    def pad(self, sequence: List[int], length: int) -> List[int]:
        """Pads a sequence to max_seq_len with a given pad_token."""
        return sequence + [self.pad_id] * (self.max_seq_len - length)

    def batch(
        self,
        sequences: List[Dict[str, List[int]]],
        batch_size: int = 8,
    ) -> List[Dict[str, torch.Tensor]]:
        """Efficiently batches tokenized data into PyTorch tensors."""
        batches = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            _size = (len(batch), self.max_seq_len)
            _input = torch.zeros(_size, dtype=torch.long)
            _target = torch.zeros(_size, dtype=torch.long)
            for j, pair in enumerate(batch):
                _input[j] = torch.tensor(pair["input"], dtype=torch.long)
                _target[j] = torch.tensor(pair["target"], dtype=torch.long)
            batches.append({"input": _input, "target": _target})
        return batches


class TextDatasetProcessor(DatasetProcessor):
    """Processor for tokenizing and batching free-form text data."""

    def tokenize(
        self,
        text: str,
        max_seq_len: int,
        add_bos: bool = True,
        add_eos: bool = True,
        batch_stride: int = 64,
    ) -> EncodedDataset:
        """Tokenizes and splits raw text into overlapping input-target pairs."""
        assert max_seq_len > 0, "max_seq_len must be greater than 0"
        assert batch_stride > 0, "batch_stride must be greater than 0"
        assert (
            batch_stride <= max_seq_len
        ), "batch_stride must be less than or equal to max_seq_len"

        pad_id = self.processor.pad_id()
        if pad_id < 0:
            pad_id = 0  # Ensure a valid pad token

        tokens = self.processor.encode(text, add_bos=add_bos, add_eos=add_eos)
        sequences = []

        if len(tokens) < max_seq_len:
            sequences.append(
                {
                    "input": self.pad_or_truncate(tokens, max_seq_len, pad_id),
                    "target": self.pad_or_truncate(tokens[1:], max_seq_len, pad_id),
                }
            )
        else:
            # num_sequences = (len(tokens) - max_seq_len) // batch_stride + 1
            for i in range(0, len(tokens) - max_seq_len, batch_stride):
                input_tokens = tokens[i : i + max_seq_len]
                target_tokens = tokens[i + 1 : i + max_seq_len + 1]  # Shifted right

                sequences.append(
                    {
                        "input": self.pad_or_truncate(
                            input_tokens, max_seq_len, pad_id
                        ),
                        "target": self.pad_or_truncate(
                            target_tokens, max_seq_len, pad_id
                        ),
                    }
                )

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
