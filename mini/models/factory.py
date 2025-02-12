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

    def __init__(self, config: ConfigTransformer):
        self.config = config

    def create_model(self) -> nn.Module:
        """
        Create a model instance based on the configuration.
        """
        model_class = MODEL_REGISTRY.get(self.config.architecture)
        if model_class is None:
            raise ValueError(
                f"Unknown model architecture: {self.config.architecture}. Available: {list(MODEL_REGISTRY.keys())}"
            )
        return model_class(self.config)
