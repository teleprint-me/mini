"""
Copyright © 2023 Austin Berrio
Module: mini.models.factory
Description: Factory module for creating model instances.
"""

import torch.nn as nn

from mini.config.transformer import ConfigTransformer
from mini.models.misty import MistyModel
from mini.models.valerie import ValerieModel

# Registry of available models
MODEL_REGISTRY = {
    "misty": MistyModel,
    "valerie": ValerieModel,
}


class ModelFactory:
    """
    Factory class for creating model instances.
    """

    DEFAULT_MODEL = "misty"

    def __init__(self, config: ConfigTransformer):
        self.config = config

    @property
    def model_name(self) -> str:
        """Returns the validated model name, falling back to the default if unspecified."""
        model_name = self.config.architecture or self.DEFAULT_MODEL
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model architecture: {model_name}. "
                f"Available: {self.list_models()}"
            )
        return model_name

    @property
    def model_class(self) -> nn.Module:
        """Returns the model class corresponding to the validated model name."""
        return MODEL_REGISTRY[self.model_name]

    def list_models(self) -> list[str]:
        """Returns a list of available model names."""
        return list(MODEL_REGISTRY.keys())

    def create_model(self) -> nn.Module:
        """Creates and returns an instance of the selected model."""
        return self.model_class(self.config)
