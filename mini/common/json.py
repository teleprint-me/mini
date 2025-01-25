"""
Copyright © 2023 Austin Berrio

Module: mini.common.json

Description: This module provides utility functions for working with JSON data.
"""

import json
from typing import Any, Dict, List, Union

import torch


class JsonUtils:
    """
    Utility class for working with JSON data.
    """

    @staticmethod
    def load_json(file_path: str) -> Any:
        """
        Load data from a JSON file.
        Args:
            file_path (str): The path to the JSON file.
        Returns:
            Any: The loaded data.
        """
        with open(file_path, "r") as f:
            data = json.load(f)
        print(f"JSON loaded from {file_path}")
        return data

    @staticmethod
    def save_json(data: Union[List, Dict], file_path: str) -> None:
        """
        Dump data to a JSON file with support for Torch types.
        Args:
            data (object): The data to be dumped.
            file_path (str): The path to the JSON file.
        Returns:
            None:
        """

        def default_serializer(obj):
            if isinstance(obj, (torch.float32, torch.float64)):
                return float(obj)  # Convert Torch floats to Python floats
            if isinstance(obj, (torch.int32, torch.int64)):
                return int(obj)  # Convert Torch integers to Python integers
            if isinstance(obj, torch.Tensor):
                return obj.tolist()  # Convert Torch arrays to lists
            raise TypeError(
                f"Object of type {type(obj).__name__} is not JSON serializable"
            )

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=default_serializer)
        print(f"JSON saved to {file_path}")

    @staticmethod
    def parse_json(json_string: str) -> Dict[str, Any]:
        """
        Parse a JSON string into a dictionary.
        Args:
            json_string (str): The JSON string to parse.
        Returns:
            Dict[str, Any]: The JSON data as a dictionary.
        """
        return json.loads(json_string)
