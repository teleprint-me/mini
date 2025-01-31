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
TensorDataset = List[Dict[str, torch.Tensor]]


class MiniProcessor:
    """Base processor providing utility functions for text processing."""

    def __init__(self, processor: SentencePieceProcessor, verbose: bool = False):
        self.processor = processor
        log_level = logging.DEBUG if verbose else logging.INFO
        self.logger = get_logger(self.__class__.__name__, log_level)

    def pad_or_truncate(
        self, tokens: List[int], max_seq_len: int, pad_id: int
    ) -> List[int]:
        """Ensures all sequences are the same length by padding or truncating."""
        return tokens[:max_seq_len] + [pad_id] * max(0, max_seq_len - len(tokens))

    def batch(
        self,
        encoded_dataset: EncodedDataset,
        batch_size: int = 8,
        dtype: torch.dtype = torch.long,
    ) -> TensorDataset:
        """Efficiently batches tokenized data into PyTorch tensors."""
        batches = []
        for i in range(0, len(encoded_dataset), batch_size):
            batch = encoded_dataset[i : i + batch_size]
            if not batch:
                continue
            batch_size_actual = len(batch)
            max_seq_len = len(batch[0]["input"])

            batch_input = torch.zeros((batch_size_actual, max_seq_len), dtype=dtype)
            batch_target = torch.zeros((batch_size_actual, max_seq_len), dtype=dtype)

            for idx, item in enumerate(batch):
                batch_input[idx] = torch.tensor(item["input"], dtype=dtype)
                batch_target[idx] = torch.tensor(item["target"], dtype=dtype)

            batches.append({"input": batch_input, "target": batch_target})
        return batches


class MiniTextProcessor(MiniProcessor):
    """Processor for tokenizing and batching free-form text data."""

    def tokenize(
        self,
        text: str,
        max_seq_len: int,
        add_bos: bool = True,
        add_eos: bool = True,
        stride: int = 64,
    ) -> EncodedDataset:
        """Tokenizes and splits raw text into overlapping input-target pairs."""
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
            for i in range(0, len(tokens) - max_seq_len, stride):
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


class MiniJsonProcessor(MiniProcessor):
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
