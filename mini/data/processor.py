"""
Copyright © 2023 Austin Berrio
Module: mini.data.processor
Description: Handles text and JSON data processing for NLP tasks.
"""

import logging
from typing import Any, Dict, List

import torch
from sentencepiece import SentencePieceProcessor

from mini.common.logger import get_logger


class DatasetProcessor:
    """Base processor providing utility functions for text processing."""

    def __init__(
        self,
        processor: SentencePieceProcessor,
        max_seq_len: int = 128,
        batch_size: int = 8,
        add_bos: bool = True,
        add_eos: bool = True,
        supervise: bool = False,
        verbose: bool = False,
    ):
        self.processor = processor
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.supervise = supervise

        self.pad_id = max(0, processor.pad_id())

        level = logging.DEBUG if verbose else logging.INFO
        self.logger = get_logger(self.__class__.__name__, level=level)

    def _pad(self, sequence: List[int]) -> List[int]:
        """Pad a sequence to max_seq_len with a given pad_id."""
        return sequence + [self.pad_id] * (self.max_seq_len - len(sequence))

    def batch(
        self, sequences: List[Dict[str, List[int]]]
    ) -> List[Dict[str, torch.Tensor]]:
        """Batch tokenized data into PyTorch tensors."""

        batches = []

        for i in range(0, len(sequences), self.batch_size):
            batch = sequences[i : i + self.batch_size]

            _size = (len(batch), self.max_seq_len)
            _input = torch.full(_size, self.pad_id, dtype=torch.long)
            _target = torch.full(_size, self.pad_id, dtype=torch.long)

            for j, pair in enumerate(batch):
                _input[j] = torch.tensor(pair["input"], dtype=torch.long)
                _target[j] = torch.tensor(pair["target"], dtype=torch.long)

            batches.append({"input": _input, "target": _target})

        return batches

    def encode(self, dataset: Any) -> List[Dict[str, List[int]]]:
        """Generate progressive input-target pairs from structured or unstructured text."""
        raise NotImplementedError("This method must be implemented by subclasses.")


class TextDatasetProcessor(DatasetProcessor):
    """Processor for tokenizing and batching free-form text data."""

    def _unsupervised(self, sequence: List[int]) -> List[Dict[str, List[int]]]:
        """Generates training pairs for unsupervised next-token prediction."""

        pairs = []

        for i in range(1, len(sequence)):
            _input = self._pad(sequence[:i])
            _target = self._pad(sequence[i : i + 1])

            if len(_input) != len(_target):
                raise ValueError(f"Sequence mismatch: {len(_input)} != {len(_target)}")

            pairs.append({"input": _input, "target": _target})

        return pairs

    def _supervised(self, sequence: List[int]) -> List[Dict[str, List[int]]]:
        """Generates training pairs for supervised next-token prediction."""

        pairs = []

        for i in range(1, len(sequence)):
            _input = self._pad(sequence[:i])
            _target = self._pad(sequence[: i + 1])

            if len(_input) != len(_target):
                raise ValueError(f"Sequence mismatch: {len(_input)} != {len(_target)}")

            pairs.append({"input": _input, "target": _target})

        return pairs

    def _next_token(self, sequence: List[int]) -> List[Dict[str, List[int]]]:
        """Wrapper for generating progressive training pairs for next-token prediction."""
        return (
            self._supervised(sequence)
            if self.supervise
            else self._unsupervised(sequence)
        )

    def encode(self, dataset: str) -> List[Dict[str, List[int]]]:
        """Generates progressive input-target pairs from text."""

        tokens = self.processor.encode(
            dataset, add_bos=self.add_bos, add_eos=self.add_eos
        )

        # Text is short, use progressive unmasking
        if len(tokens) <= self.max_seq_len:
            return self._next_token(tokens)

        # Text is long, chunk it into max_seq_len-sized pieces
        pairs = []
        for i in range(0, len(tokens), self.max_seq_len):
            sequence = tokens[i : i + self.max_seq_len]
            block = self._next_token(sequence)
            pairs.extend(block)

        return pairs


class JsonDatasetProcessor(DatasetProcessor):
    """Processor for tokenizing and batching structured JSON data."""

    def encode(self, dataset: List[Dict[str, str]]) -> List[Dict[str, List[int]]]:
        """Tokenizes structured instruction-response pairs."""
        sequences = []
        for entry in dataset:
            user = entry.get("user", "").strip()
            assistant = entry.get("assistant", "").strip()

            if not user:
                raise ValueError(f"Missing user data: {entry}")
            if not assistant:
                raise ValueError(f"Missing assistant data: {entry}")

            _input = self.processor.encode(user, add_bos=self.add_bos, add_eos=False)
            _target = self.processor.encode(
                assistant, add_bos=False, add_eos=self.add_eos
            )

            sequences.append({"input": self._pad(_input), "target": self._pad(_target)})
        return sequences
