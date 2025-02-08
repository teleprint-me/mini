"""
Copyright © 2023 Austin Berrio
Module: mini.config.base
Description: Base configuration class for mini projects.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ConfigBase:
    """Base class for common configuration utilities."""

    def as_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the config."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
