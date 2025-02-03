"""
Module: mini.transformer.optimizer
Description: Defines a class for creating and managing optimizers in the transformer model.
"""

from typing import Optional

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler


class MiniOptimizer:
    @staticmethod
    def create_optimizer(model: nn.Module, **kwargs) -> optim.Optimizer:
        """Creates an optimizer based on the given configuration."""
        optimizer_type = kwargs.get("optimizer", "adam").lower()
        optimizer_params = {
            "lr": kwargs.get("lr", 1e-3),
            "eps": kwargs.get("eps", 1e-8),
            "weight_decay": kwargs.get("weight_decay", 0),
        }

        if optimizer_type == "adam":
            return optim.Adam(
                model.parameters(),
                **optimizer_params,
                amsgrad=kwargs.get("amsgrad", False),
            )
        elif optimizer_type == "adamw":
            return optim.AdamW(
                model.parameters(),
                **optimizer_params,
                amsgrad=kwargs.get("amsgrad", False),
            )
        elif optimizer_type == "sgd":
            return optim.SGD(
                model.parameters(),
                **optimizer_params,
                momentum=kwargs.get("momentum", 0.9),
                dampening=kwargs.get("dampening", 0),
                nesterov=kwargs.get("nesterov", False),
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    @staticmethod
    def create_scheduler(optimizer: optim.Optimizer, **kwargs) -> Optional[LRScheduler]:
        """Creates an LR scheduler for the optimizer."""
        scheduler_type = kwargs.get("scheduler", "step").lower()

        if scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get("step_size", 10),
                gamma=kwargs.get("gamma", 0.1),
            )
        elif scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get("t_max", 50),
                eta_min=kwargs.get("eta_min", 1e-6),
            )
        elif scheduler_type == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=kwargs.get("start_factor", 0.1),
                total_iters=kwargs.get("total_iters", 100),
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")

    @staticmethod
    def create_criterion(
        criterion_type: str = "cross_entropy", pad_id: Optional[int] = None
    ) -> nn.Module:
        """Creates a loss function based on the given criterion type."""
        criterion_type = criterion_type.lower()

        if criterion_type == "cross_entropy":
            return (
                nn.CrossEntropyLoss(ignore_index=pad_id)
                if pad_id is not None
                else nn.CrossEntropyLoss()
            )
        elif criterion_type == "mse":
            return nn.MSELoss()
        elif criterion_type == "mae":
            return nn.L1Loss()
        else:
            raise ValueError(f"Unsupported criterion: {criterion_type}")
