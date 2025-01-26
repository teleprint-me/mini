"""
Copyright © 2023 Austin Berrio

Module: mini.common.json

Description: This module provides utility functions for working with JSON data.
"""

import json
import logging
import sys
from typing import Any

import jsonschema
import torch

from mini.common.logger import get_logger


class JsonUtils:
    """
    Utility class for working with JSON data.
    """

    def __init__(self, verbose: bool = False):
        log_level = logging.DEBUG if verbose else logging.INFO
        self.logger = get_logger(name=self.__class__.__name__, level=log_level)

    def validate_json(self, data: Any, schema: dict = None) -> bool:
        """
        Validate JSON data against a schema.
        Args:
            data (Any): The JSON data to validate.
            schema (dict, optional): The JSON schema to validate against. If None, validation is skipped.
        Returns:
            bool: True if the data is valid or if no schema is provided, False otherwise.
        """
        if not schema:
            self.logger.warning("No schema provided. Skipping validation.")
            return True

        try:
            jsonschema.validate(instance=data, schema=schema)
            self.logger.debug("JSON data is valid.")
            return True
        except jsonschema.exceptions.ValidationError as e:
            self.logger.error(f"JSON data is invalid: {e}")
            sys.exit(1)
        except jsonschema.exceptions.SchemaError as e:
            self.logger.critical(f"Schema error: {e}")
            sys.exit(1)

    def load_json(self, file_path: str) -> Any:
        """
        Load data from a JSON file.
        Args:
            file_path (str): The path to the JSON file.
        Returns:
            Any: The loaded data.
        """
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                self.logger.debug(f"JSON loaded from {file_path}")
                return data
        except FileNotFoundError as e:
            self.logger.error(f"Error loading JSON from {file_path}: {e}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON from {file_path}: {e}")
            sys.exit(1)

    def save_json(self, file_path: str, data: Any) -> None:
        """
        Dump data to a JSON file with support for Torch types.
        Args:
            data (object): The data to be dumped.
            file_path (str): The path to the JSON file.
        Returns:
            None:
        """

        def default_serializer(obj: object) -> Any:
            # Handle Python float and integer types directly
            if isinstance(obj, (float, int)):
                return obj
            # Handle PyTorch dtypes explicitly if needed
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            # Handle PyTorch dtypes explicitly if needed
            if isinstance(obj, torch.dtype):
                return str(obj)
            raise TypeError(
                f"Object of type {type(obj).__name__} is not JSON serializable"
            )

        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2, default=default_serializer)
            self.logger.debug(f"JSON saved to {file_path} with {len(data)} items.")
        except TypeError as e:
            self.logger.error(f"Error decoding JSON from {file_path}: {e}")
            sys.exit(1)

        def parse_json(self, json_string: str) -> Any:
            """
            Parse a JSON string into a dictionary.
            Args:
                json_string (str): The JSON string to parse.
            Returns:
                Dict[str, Any]: The JSON data as a dictionary.
            """
            try:
                data = json.loads(json_string)
                self.logger.debug(f"JSON parsed from string: {data}")
                return data
            except json.JSONDecodeError as e:
                self.logger.error(f"Error decoding JSON: {e}")
                sys.exit(1)
