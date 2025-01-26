"""
Copyright © 2023 Austin Berrio

Module: mini.data.processor

Description: This module provides functions to load and process datasets for NLP models.
"""

import logging
from typing import Dict, List

import torch
from sentencepiece import SentencePieceProcessor

from mini.common.json import JsonUtils
from mini.common.logger import get_logger

# Dataset type annotations
JsonInstruction = str
JsonResponse = str
JsonInstructionResponsePair = Dict[JsonInstruction, JsonResponse]
JsonDataset = List[JsonInstructionResponsePair]
# Encoding type annotations
Encoding = int
EncodedInstruction = List[int]
EncodedResponse = List[int]
EncodedInstructionResponsePair = Dict[EncodedInstruction, EncodedResponse]
EncodedDataset = List[EncodedInstructionResponsePair]
# Tensor type annotations
TensorKey = str
TensorValue = torch.Tensor
TensorPair = Dict[TensorKey, TensorValue]
TensorDataset = List[TensorPair]


# Define the MiniDataProcessor class
class MiniDataProcessor:
    def __init__(self, processor: SentencePieceProcessor, verbose: bool = False):
        self.processor = processor
        log_level = logging.DEBUG if verbose else logging.INFO
        self.logger = get_logger(self.__class__.__name__, log_level)

    def tokenize(
        self,
        json_dataset: JsonDataset,
        max_length: int = 256,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> EncodedDataset:
        """
        Tokenize the dataset into instruction-response pairs, retaining separate fields for input and target tokens.
        Args:
            json_dataset (JsonDataset): Dataset of instruction-response pairs. Each entry should have 'instruction' and 'response' keys.
            max_length (int): Maximum length for tokenized sequences.
            add_bos (bool): Add a beginning-of-sequence token.
            add_eos (bool): Add an end-of-sequence token.
        Returns:
            EncodedDataset: Tokenized dataset with input and target token fields.
        """
        encoded_dataset = []
        pad_id = self.processor.pad_id()

        for entry in json_dataset:
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

            encoded_dataset.append(
                {
                    "input": input_tokens,
                    "target": target_tokens,
                }
            )

        return encoded_dataset

    def batch(
        self,
        encoded_dataset: EncodedDataset,
        batch_size: int = 8,
        dtype: torch.dtype = torch.int,
        device: torch.device = None,
    ) -> TensorDataset:
        """
        Create batches of tokenized data for model input.
        Args:
            encoded_dataset (EncodedDataset): Encoded dataset with "input" and "target" fields.
            batch_size (int): Number of samples per batch.
            device (torch.device): Device to place the tensors on. Default is None (CPU).
            dtype (torch.dtype): Data type for the batch tensors. Default is torch.int.
        Raises:
            ValueError: If batch_size is less than or equal to 0.
        Returns:
            TensorDataset: Batches of tokenized data with torch tensors.
        """
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        batches = []
        for i in range(0, len(encoded_dataset), batch_size):
            batch = encoded_dataset[i : i + batch_size]
            batch_dict = {
                key: torch.tensor(
                    [item[key] for item in batch], dtype=dtype, device=device
                )
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
        "--schema-path",
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
        default=8,
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

    # Load JSON schema if provided
    json_utils = JsonUtils(verbose=args.verbose)
    json_dataset = json_utils.load_json(args.input)

    # Load schema from file if provided
    schema = None
    if args.schema_path:
        schema = json_utils.load_json(args.schema_path)

    # Validate JSON data if provided schema
    json_utils.validate_json(json_dataset, schema=schema)

    # Initialize SentencePieceProcessor and MiniDataProcessor
    processor = SentencePieceProcessor(args.model_path)
    mini_data_processor = MiniDataProcessor(processor=processor, verbose=args.verbose)

    # Tokenize and batch the json_dataset
    tokenized_dataset = mini_data_processor.tokenize(
        json_dataset, args.max_length, add_bos=args.add_bos, add_eos=args.add_eos
    )
    batched_dataset = mini_data_processor.batch(tokenized_dataset, args.batch_size)

    # Save the batched dataset to a file
    json_utils.save_json(args.output, batched_dataset)
    print("Dataset processing completed successfully.")
