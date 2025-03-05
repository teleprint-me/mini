"""
Copyright © 2023 Austin Berrio
Module: mini.data.loader
Description: Automate loading datasets for text-to-text sequencing.
"""

import logging
import random

import torch
from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset

from mini.common.json import JsonUtils
from mini.common.logger import get_logger
from mini.data.processor import JsonDatasetProcessor, TextDatasetProcessor


class DatasetLoader(Dataset):
    """A dataset class for loading and processing text data."""

    def __init__(
        self,
        file_path: str,
        processor: SentencePieceProcessor,
        max_seq_len: int = 128,
        batch_size: int = 8,
        add_bos: bool = True,
        add_eos: bool = True,
        supervise: bool = False,
        verbose: bool = False,
    ):
        self.file_path = file_path
        self.processor = processor
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.supervise = supervise
        self.verbose = verbose

        self.batches = []

        level = logging.DEBUG if self.verbose else logging.INFO
        self.logger = get_logger(self.__class__.__name__, level=level)
        self.logger.info(f"Initializing dataset from {self.file_path}")

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a tokenized input-target pair as torch tensors."""
        # Each sequence represents an individually sampled sequence pair, e.g.
        # idx: 2, input: ['The', 'quick', 0, ...], target: ['The', 'quick', 'brown', ...]
        batch_idx = idx // self.batch_size
        sample_idx = idx % self.batch_size
        batch = self.batches[batch_idx]
        return batch["input"][sample_idx], batch["target"][sample_idx]

    def __len__(self) -> int:
        """Return the number of batches in the dataset."""
        return len(self.batches)

    def __iter__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Make Dataset an iterator by returning an iterable."""
        for batch in self.batches:
            yield batch["input"], batch["target"]

    def load_data(self):
        """Load and process data into batches."""
        raise NotImplementedError("This method must be implemented by subclasses.")


class TextDatasetLoader(DatasetLoader):
    """Custom dataset class for loading tokenized pre-training data."""

    def __init__(
        self,
        file_path: str,
        processor: SentencePieceProcessor,
        max_seq_len: int = 128,
        batch_size: int = 8,
        add_bos: bool = True,
        add_eos: bool = True,
        supervise: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            file_path,
            processor,
            max_seq_len,
            batch_size,
            add_bos,
            add_eos,
            supervise,
            verbose,
        )

        self.text = TextDatasetProcessor(
            self.processor,
            self.max_seq_len,
            self.batch_size,
            self.add_bos,
            self.add_eos,
            self.supervise,
            self.verbose,
        )

        self.load_data()

    def _read_file(self):
        self.logger.info(f"Loading plaintext dataset from {self.file_path}")

        with open(self.file_path, "r", encoding="utf-8") as file:
            plaintext = file.read()

        self.logger.info(f"Loaded plaintext dataset with {len(plaintext)} characters")
        assert len(plaintext) > 0, "File is empty."

        return plaintext

    def _process_batches(self, plaintext: str):
        sequences = self.text.encode(plaintext)
        batches = self.text.batch(sequences)

        self.logger.info(f"Generated {len(sequences)} progressive sequences")
        self.logger.info(f"Generated {len(batches)} training batches")
        assert len(batches) > 0, "Failed to generate batches."

        return batches

    def _log_shapes(self):
        x = self.batches[0]["input"].shape
        y = self.batches[0]["target"].shape
        self.logger.info(f"Input shape: {x}, Target shape: {y}")

    def load_data(self):
        """Load and process the text data into batches."""
        plaintext = self._read_file()
        self.batches = self._process_batches(plaintext)
        self._log_shapes()


class JsonDatasetLoader(DatasetLoader):
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
        self.schema_path = schema_path
        self.load_data()
        assert len(self.batches) > 0, "Dataset is empty"

    def load_data(self):
        """Load and preprocess JSON data."""
        self.logger.info(f"Loading JSON data from {self.file_path}")
        self.json_utils = JsonUtils(self.verbose)
        raw_dataset = self.json_utils.load_json(self.file_path)
        self.logger.debug(f"Loaded JSON dataset with {len(raw_dataset)} records")

        # Validate dataset
        if self.schema_path:
            self.logger.info(f"Validating JSON data against schema {self.schema_path}")
            raw_schema = self.json_utils.load_json(self.schema_path)
            self.json_utils.validate_json(raw_dataset, raw_schema)
            self.logger.debug("Dataset successfully validated against schema")

        # Shuffle dataset
        random.shuffle(raw_dataset)
        self.logger.debug("Shuffled dataset for training")

        # Use JsonDatasetProcessor for tokenization and batching
        self.json_processor = JsonDatasetProcessor(self.processor, self.verbose)
        self.encoded = self.json_processor.tokenize(raw_dataset, self.max_seq_len)
        self.batches = self.json_processor.batch(self.encoded, self.batch_size)
        self.logger.debug(f"Generated {len(self.batches)} training batches")
