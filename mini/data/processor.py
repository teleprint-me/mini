"""
Copyright © 2023 Austin Berrio
Module: mini.data.processor
Description: This module provides functions to load and process datasets for NLP models.
"""

import logging
from typing import Dict, List

import torch
from sentencepiece import SentencePieceProcessor

from mini.common.logger import get_logger

# Dataset type annotations
JsonInput = str
JsonTarget = str
JsonPair = Dict[JsonInput, JsonTarget]
JsonDataset = List[JsonPair]
# Encoding type annotations
Encoding = int
EncodedInput = List[Encoding]
EncodedTarget = List[Encoding]
EncodedPair = Dict[EncodedInput, EncodedTarget]
EncodedDataset = List[EncodedPair]
# Tensor type annotations
TensorKey = str
TensorValue = torch.Tensor
TensorPair = Dict[TensorKey, TensorValue]
TensorDataset = List[TensorPair]


class MiniProcessor:
    def __init__(self, processor: SentencePieceProcessor, verbose: bool = False):
        self.processor = processor
        log_level = logging.DEBUG if verbose else logging.INFO
        self.logger = get_logger(self.__class__.__name__, log_level)

    def pad_or_truncate(
        self, tokens: List[int], max_seq_len: int, pad_id: int
    ) -> List[int]:
        """Helper function to pad or truncate token sequences."""
        return tokens[:max_seq_len] + [pad_id] * max(0, max_seq_len - len(tokens))

    def batch(
        self,
        encoded_dataset: EncodedDataset,
        batch_size: int = 8,
        dtype: torch.dtype = torch.long,
    ) -> TensorDataset:
        """
        Create batches of tokenized data for model input.
        Args:
            encoded_dataset (EncodedDataset): Encoded dataset with "input" and "target" fields.
            batch_size (int): Number of samples per batch.
            dtype (torch.dtype): Data type for the batch tensors. Default is torch.long.
        Returns:
            TensorDataset: Batches of tokenized data with torch tensors.
        """
        batches = []
        for i in range(0, len(encoded_dataset), batch_size):
            batch = encoded_dataset[i : i + batch_size]
            if not batch:  # Skip empty batches
                continue
            batch_dict = {
                key: torch.tensor([item[key] for item in batch], dtype=dtype)
                for key in batch[0].keys()
            }
            batches.append(batch_dict)
        return batches


class MiniTextProcessor(MiniProcessor):
    def __init__(self, processor: SentencePieceProcessor, verbose: bool = False):
        super().__init__(processor, verbose)

    def tokenize(
        self,
        text: str,
        max_seq_len: int,
        add_bos: bool = True,
        add_eos: bool = True,
        stride: int = 64,
    ) -> EncodedDataset:
        """
        Tokenize and split text into overlapping input-target pairs.
        Args:
            text (str): Raw input text.
            max_seq_len (int): Maximum sequence length.
            stride (int): Overlap for consecutive segments (default: 64).
        Returns:
            List[Dict[str, List[int]]]: Tokenized input-target pairs.
        """
        pad_id = self.processor.pad_id()
        if pad_id < 0:
            pad_id = 0  # Ensure a valid pad token

        tokens = self.processor.encode(text, add_bos=add_bos, add_eos=add_eos)
        sequences = []

        for i in range(0, len(tokens) - max_seq_len, stride):  # Sliding window
            input_tokens = tokens[i : i + max_seq_len]
            target_tokens = tokens[i + 1 : i + max_seq_len + 1]  # Shifted right

            sequences.append(
                {
                    "input": self.pad_or_truncate(input_tokens, max_seq_len, pad_id),
                    "target": self.pad_or_truncate(target_tokens, max_seq_len, pad_id),
                }
            )

        return sequences


# Define the MiniJsonProcessor class
class MiniJsonProcessor(MiniProcessor):
    def __init__(self, processor: SentencePieceProcessor, verbose: bool = False):
        super().__init__(processor, verbose)

    def tokenize(
        self,
        json_dataset: JsonDataset,
        max_seq_len: int,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> EncodedDataset:
        """
        Tokenize the dataset into instruction-response pairs, retaining separate fields for input and target tokens.
        Args:
            json_dataset (JsonDataset): Dataset of instruction-response pairs. Each entry should have 'instruction' and 'response' keys.
            max_seq_len (int): Maximum length for tokenized sequences.
            add_bos (bool): Add a beginning-of-sequence token.
            add_eos (bool): Add an end-of-sequence token.
        Returns:
            EncodedDataset: Tokenized dataset with input and target token fields.
        """
        encoded_dataset = []
        pad_id = self.processor.pad_id()
        pad_id = pad_id if pad_id > -1 else 0

        for entry in json_dataset:
            instruction = entry.get("instruction", "")
            response = entry.get("response", "")

            input_tokens = self.processor.encode(
                instruction, add_bos=add_bos, add_eos=False
            )
            target_tokens = self.processor.encode(
                response, add_bos=False, add_eos=add_eos
            )

            encoded_dataset.append(
                {
                    "input": self.pad_or_truncate(input_tokens, max_seq_len, pad_id),
                    "target": self.pad_or_truncate(target_tokens, max_seq_len, pad_id),
                }
            )

        return encoded_dataset
