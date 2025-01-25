"""
Module: mini.data.processor
Description: This module provides functions to load and process datasets for NLP models.
"""

import json
import logging
from typing import Any, Dict, List, Union

import torch
from jsonschema import validate
from sentencepiece import SentencePieceProcessor

from mini.common.logger import get_logger
from mini.data.chunker import MiniDataChunker

# NOTE: Not sure if this is a good idea. This is pretty complicated.
Label = int
Tokens = List[int]
Dataset = List[Dict[str, Union[int, str]]]
TokenizedDataset = List[Dict[str, Union[Tokens, Label]]]


class MiniDataProcessor:
    def __init__(self, processor: SentencePieceProcessor, verbose: bool = False):
        self.processor = processor
        self.chunker = MiniDataChunker(self.processor, verbose=verbose)
        log_level = logging.DEBUG if verbose else logging.INFO
        self.logger = get_logger(self.__class__.__name__, log_level)

    def load(self, file_path: str, schema: Dict[str, Any]) -> Any:
        with open(file_path, "r") as f:
            try:
                dataset = json.load(f)
                validate(instance=dataset, schema=schema)
                self.logger.info(f"Dataset loaded successfully: {file_path}")
                return dataset
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.error(f"Error loading dataset: {e}")
                return []

    def save(
        self,
        file_path: str,
        dataset: List[Dict[str, Union[torch.Tensor, List]]],
    ) -> None:
        with open(file_path, "w") as f:

            def default(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.tolist()
                return obj

            json.dump(dataset, f, indent=2, default=default)
        self.logger.info(f"Dataset saved to {file_path}")

    def tokenize(
        self,
        dataset: Dataset,
        max_length: int = 256,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> TokenizedDataset:
        """
        Tokenize the dataset into instruction-response pairs, retaining separate fields for input and target tokens.

        Args:
            dataset (Dataset): The dataset to tokenize. Assumes each entry has "instruction" and "response" fields.
            max_length (int): Maximum length for tokenized sequences.
            add_bos (bool): Add a beginning-of-sequence token.
            add_eos (bool): Add an end-of-sequence token.

        Returns:
            TokenizedDataset: Tokenized dataset with input and target token fields.
        """
        tokenized_data = []
        pad_id = self.processor.pad_id()

        for entry in dataset:
            instruction = entry.get("instruction", "")
            response = entry.get("response", "")

            # Tokenize instruction
            input_tokens = self.processor.encode(
                instruction, add_bos=add_bos, add_eos=False
            )

            # Tokenize response
            target_tokens = self.processor.encode(
                response, add_bos=False, add_eos=add_eos
            )

            # Truncate and pad input tokens
            input_tokens = input_tokens[:max_length] + [pad_id] * max(
                0, max_length - len(input_tokens)
            )

            # Truncate and pad target tokens
            target_tokens = target_tokens[:max_length] + [pad_id] * max(
                0, max_length - len(target_tokens)
            )

            tokenized_data.append(
                {
                    "input": input_tokens,
                    "target": target_tokens,
                }
            )

        return tokenized_data

    def batch(
        self,
        tokenized_data: List[Dict[str, Union[int, List[int]]]],
        batch_size: int = 32,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Create batches of tokenized data for model input.

        Args:
            tokenized_data (List[Dict[str, Union[int, List[int]]]]): Tokenized dataset with "input" and "target" fields.
            batch_size (int): Number of samples per batch.

        Returns:
            List[Dict[str, torch.Tensor]]: Batches of tokenized data with tensors.
        """
        batches = []
        for i in range(0, len(tokenized_data), batch_size):
            batch = tokenized_data[i : i + batch_size]
            batch_dict = {
                key: torch.tensor([item[key] for item in batch], dtype=torch.long)
                for key in batch[0].keys()
            }
            batches.append(batch_dict)
        return batches


# Usage example:
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process dataset and tokenize it.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the tokenizer model file.",
    )
    parser.add_argument(
        "--schema",
        type=str,
        required=True,
        help="Path to the schema file for the dataset.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input dataset file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output tokenized dataset file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for tokenization.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum length for tokenization.",
    )
    parser.add_argument(
        "--add-bos",
        action="store_true",
        help="Add beginning-of-sentence token.",
    )
    parser.add_argument(
        "--add-eos",
        action="store_true",
        help="Add end-of-sentence token.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )
    args = parser.parse_args()

    processor = SentencePieceProcessor(args.model_path)
    mini_data_processor = MiniDataProcessor(processor=processor, verbose=args.verbose)
    dataset = mini_data_processor.load(args.input, args.schema)

    tokenized_dataset = mini_data_processor.tokenize(
        dataset, args.max_length, add_bos=args.add_bos, add_eos=args.add_eos
    )
    batched_dataset = mini_data_processor.batch(tokenized_dataset, args.batch_size)
    mini_data_processor.save(args.output, batched_dataset)

    print("Dataset processing completed successfully.")
