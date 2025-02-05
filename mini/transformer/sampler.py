"""
Copyright © 2023 Austin Berrio
Module: mini.transformer.sampler
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

from mini.common.logger import get_logger


@dataclass
class SamplerConfig:
    top_k: int = 50
    top_p: float = 0.9
    temperature: float = 0.8
    repetition_penalty: float = 1.2
    greedy: bool = False  # Greedy decoding mode
    pad_id: Optional[int] = None
    verbose: bool = False  # Enable debug mode


class MiniSampler:
    def __init__(self, config: SamplerConfig):
        self.config = config
        self.logger = get_logger(
            name=self.__class__.__name__,
            level=logging.DEBUG if config.verbose else logging.INFO,
        )

    def temperature(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling and clip extreme values for numerical stability."""
        if self.config.temperature > 0:
            logits = logits / self.config.temperature
            logits = torch.clamp(logits, min=-10, max=10)  # Prevent extreme values
            self.logger.debug(f"Logits after temperature scaling: {logits}")
        return logits

    def repetition_penalty(
        self, logits: torch.Tensor, past_tokens: List[int]
    ) -> torch.Tensor:
        """Apply a repetition penalty, making penalties stronger for more recent tokens."""
        if past_tokens:
            for i, token in enumerate(
                reversed(past_tokens)
            ):  # Penalize recent tokens more
                penalty = self.config.repetition_penalty ** ((i + 1) / len(past_tokens))
                logits[:, token] /= penalty
            self.logger.debug(f"Logits after repetition penalty: {logits}")
        return logits

    def softmax(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply softmax to convert logits to probabilities."""
        probs = F.softmax(logits, dim=-1)
        self.logger.debug(f"Softmax probabilities: {probs}")
        return probs

    def top_k(self, probs: torch.Tensor) -> torch.Tensor:
        """Apply top-k filtering."""
        if self.config.top_k > 0:
            values, _ = torch.topk(probs, self.config.top_k)
            min_prob = values[:, -1].unsqueeze(-1)
            probs = torch.where(probs < min_prob, torch.zeros_like(probs), probs)
            self.logger.debug(f"Probabilities after top-k filtering: {probs}")
        return probs  # No filtering if top_k = 0

    def top_p(self, probs: torch.Tensor) -> torch.Tensor:
        """Apply nucleus (top-p) filtering with stability improvements."""
        sorted_probs, indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        mask = cumulative_probs > self.config.top_p
        sorted_probs[mask] = 0

        sorted_probs.div_(
            sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        )  # Normalize safely
        probs.scatter_(-1, indices, sorted_probs)

        self.logger.debug(f"Probabilities after top-p filtering: {probs}")
        return probs

    def sample(self, logits: torch.Tensor, past_tokens: List[int]) -> int:
        """Perform sampling using configured strategies."""
        self.logger.debug(f"Initial logits: {logits}")

        logits = self.temperature(logits)
        logits = self.repetition_penalty(logits, past_tokens)

        if self.config.pad_id is not None:
            logits[:, self.config.pad_id] = -float("inf")  # Mask padding token
            self.logger.debug(
                f"Logits after masking pad_id {self.config.pad_id}: {logits}"
            )

        probs = self.softmax(logits)
        probs = self.top_k(probs)
        probs = self.top_p(probs)

        if self.config.greedy:
            selected_token = torch.argmax(probs, dim=-1).item()
            self.logger.debug(f"Greedy selected token: {selected_token}")
        else:
            selected_token = torch.multinomial(probs, 1).item()
            self.logger.debug(f"Sampled token: {selected_token}")

        return selected_token
