"""
Copyright © 2023 Austin Berrio
Module: mini.data.set
Description: Automate loading datasets for text-to-text sequencing.
"""

import logging
import random

import torch
from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset

from mini.common.json import JsonUtils
from mini.common.logger import get_logger
from mini.data.processor import MiniJsonProcessor, MiniTextProcessor


class MiniDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        processor: SentencePieceProcessor,
        max_seq_len: int,
        batch_size: int = 8,
        verbose: bool = False,
    ):
        self.file_path = file_path
        self.processor = processor
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        log_level = logging.DEBUG if verbose else logging.INFO
        self.logger = get_logger(self.__class__.__name__, log_level)
        self.logger.info(f"Initializing dataset from {file_path}")

    def __getitem__(self, idx):
        """Return a tokenized input-target pair as torch tensors."""
        batch = self.batches[idx]
        return batch["input"], batch["target"]


class MiniTextDataset(MiniDataset):
    """Custom dataset class for loading tokenized pre-training data."""

    def __init__(
        self,
        file_path: str,
        processor: SentencePieceProcessor,
        max_seq_len: int,
        batch_size: int = 8,
        batch_stride: int = 32,
        verbose: bool = False,
    ):
        super().__init__(file_path, processor, max_seq_len, batch_size, verbose)
        # Load document as text
        with open(self.file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        self.logger.debug(f"Loaded text file with {len(raw_text)} characters")

        # Use MiniTextProcessor for tokenization and batching
        self.text_processor = MiniTextProcessor(self.processor, verbose)
        self.encoded = self.text_processor.tokenize(
            raw_text, max_seq_len=max_seq_len, batch_stride=batch_stride
        )
        self.batches = self.text_processor.batch(self.encoded, batch_size)

        self.logger.debug(f"Generated {len(self.batches)} training batches")

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Make Dataset an iterator by returning an iterable."""
        for batch in self.batches:
            yield batch["input"], batch["target"]


class MiniJsonDataset(MiniDataset):
    """Custom dataset class for loading tokenized instruction-response pairs."""

    def __init__(
        self,
        file_path: str,
        processor: SentencePieceProcessor,
        max_seq_len: int,
        batch_size: int = 8,
        schema_path: str = None,
        verbose: bool = False,
    ):
        super().__init__(file_path, processor, max_seq_len, batch_size, verbose)
        self.json_utils = JsonUtils(verbose)

        # Load dataset
        raw_dataset = self.json_utils.load_json(self.file_path)
        self.logger.debug(f"Loaded JSON dataset with {len(raw_dataset)} records")

        # Validate dataset
        if schema_path is not None:
            raw_schema = self.json_utils.load_json(schema_path)
            self.json_utils.validate_json(raw_dataset, raw_schema)
            self.logger.debug("Dataset successfully validated against schema")

        # Shuffle dataset
        random.shuffle(raw_dataset)
        self.logger.debug("Shuffled dataset for training")

        # Use MiniJsonProcessor for tokenization and batching
        self.data_processor = MiniJsonProcessor(self.processor, verbose)
        self.encoded = self.data_processor.tokenize(raw_dataset, self.max_seq_len)
        self.batches = self.data_processor.batch(self.encoded, self.batch_size)

        self.logger.debug(f"Generated {len(self.batches)} training batches")

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Make Dataset an iterator by returning an iterable."""
        for batch in self.batches:
            yield batch["input"], batch["target"]
