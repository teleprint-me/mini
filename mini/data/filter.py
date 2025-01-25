"""
Module: mini.data.filter

Description: This module provides functions for filtering and processing datasets.

Example Usage:
python strip.py --input data/gpt4-instruct-dedupe-only-dataset.json \
    --output data/gpt4-instruct-mini-dataset.json \
    --pairs 10 --shuffle --seed 42
"""

import argparse
import json
import random

import jsonschema
from tqdm import tqdm

from mini.common.json import JsonUtils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a dataset.")
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Path to the input dataset file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Path to the output dataset file.",
    )
    parser.add_argument(
        "-s",
        "--schema-output",
        required=True,
        type=str,
        help="Path to the JSON schema for the dataset.",
    )
    parser.add_argument(
        "-n",
        "--pairs",
        type=int,
        help="Number of pairs to keep in the output dataset.",
    )
    parser.add_argument(
        "-z",
        "--seed",
        type=int,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "-r",
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset before processing.",
    )
    args = parser.parse_args()

    json_utils = JsonUtils()

    # Load the dataset and process it as needed.
    input_data = json_utils.load_json(args.input)
    if args.pairs is not None:
        input_data = input_data[: args.pairs]
        if args.pairs > len(input_data):
            print(
                f"Warning: Requested {args.pairs} pairs, but only {len(input_data)} available."
            )
        print(f"Loaded {len(input_data)} items from {args.input}")

    # Set the seed
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Set random seed to {args.seed}")

    # Shuffle the data to ensure randomness.
    if args.shuffle:
        random.shuffle(input_data)
        print(f"Shuffled {len(input_data)} items")

    # Process each item in the dataset and save the output.
    output_data = []
    for item in tqdm(input_data):
        # Each item has a instruction, input, and response key.
        # Combine the instruction and input into a single instruction.
        instruction = item.get("instruction", "")
        input_field = item.get("input", "")
        instruction += f" {input_field}" if input_field else ""
        # Get the response from the item.
        response = item.get("response", "")
        if not response:
            print(f"Warning: No response found for item {json.dumps(item, indent=4)}")
            continue
        # Append the instruction and response to the output data.
        output_data.append({"instruction": instruction, "response": response})

    # Generate a JSON schema for the output data.
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "instruction": {"type": "string"},
                "response": {"type": "string"},
            },
            "required": ["instruction", "response"],
        },
    }
    # Validate the output data against the schema.
    jsonschema.validate(instance=output_data, schema=schema)
    print("Output data validated against the schema")

    # Save the schema data to a file.
    json_utils.save_json(args.schema_output, schema)
    print(f"Schema processed and saved to {args.schema_output}")

    # Save the output data to a file.
    json_utils.save_json(args.output, output_data)
    print(f"Dataset processed and saved to {args.output}")
