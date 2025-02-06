"""
Copyright © 2023 Austin Berrio
Script: mini.transformer.generator
Description: Simple completions for text-to-text generation with streaming output.
"""

import functools
import sys
from dataclasses import dataclass
from typing import List, Optional, Union

import regex as re  # Use `regex`, not `re`
import torch
from sentencepiece import SentencePieceProcessor

from mini.transformer.model import MiniRuntime
from mini.transformer.sampler import MiniSampler
from mini.transformer.state import MiniState

# Default GPT-2 style pre-tokenizer regex
DEFAULT_PRETOKENIZER = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    re.IGNORECASE,
)


@dataclass
class GeneratorConfig:
    state: MiniState
    sampler: MiniSampler
    runtime: MiniRuntime
    processor: SentencePieceProcessor
    pre_tokenizer: Optional[re.Pattern] = None


class MiniGenerator:
    def __init__(self, config: GeneratorConfig):
        self.config = config

    def __post_init__(self):
        self.load()

    @functools.cached_property
    def pad_id(self) -> int:
        return max(self.config.processor.pad_id(), 0)

    @functools.cached_property
    def eos_id(self) -> int:
        return self.config.processor.eos_id()

    @functools.cached_property
    def max_seq_len(self) -> int:
        return self.config.state.model.max_seq_len

    @functools.cached_property
    def device(self) -> torch.device:
        return self.config.runtime.device_type

    def load(self) -> None:
        """Load the model state and set to evaluation mode."""
        self.config.state.load(train=False)
        self.config.state.model.eval()

    def encode(self, prompt: str) -> List[int]:
        return self.config.processor.encode(prompt, add_bos=True, add_eos=False)

    def decode(self, encodings: Union[int, List[int]]) -> str:
        return self.config.processor.decode(encodings)

    def batch(self, encodings: List[int]) -> torch.Tensor:
        return torch.tensor([encodings], dtype=torch.long, device=self.device)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.config.state.model(x)[:, -1, :]

    def sample(self, x: torch.Tensor, past_tokens: List[int]) -> int:
        return self.config.sampler.sample(x, past_tokens=past_tokens)

    def cat(self, x: torch.Tensor, token: int) -> torch.Tensor:
        return torch.cat([x, torch.tensor([[token]], device=self.device)], dim=1)

    def pre_tokenize(self, text: str) -> List[str]:
        # Fallback to returning text as-is if no tokenizer is provided
        if not self.config.pre_tokenizer:
            return [text]
        return re.findall(self.config.pre_tokenizer, text)

    def stream(self, prompt: str, max_tokens: Optional[int] = None):
        """Generates text from the given prompt with token streaming (yields tokens)."""

        # Encode input
        input_ids = self.encode(prompt)
        input_tensor = self.batch(input_ids)

        # Token tracking
        buffer = input_ids[:]
        last_output = self.decode(input_ids[:])

        with torch.no_grad():
            max_len = max_tokens if max_tokens else self.max_seq_len
            for _ in range(max_len - len(buffer)):
                logits = self.predict(input_tensor)
                next_token = self.sample(logits, buffer)

                if next_token == self.pad_id:
                    continue  # Ignore PAD token

                buffer.append(next_token)
                input_tensor = self.cat(input_tensor, next_token)

                # Decode with full context
                new_output = self.decode(buffer)

                # Use the hot-swappable pre-tokenizer
                old_tokens = self.pre_tokenize(last_output)
                new_tokens = self.pre_tokenize(new_output)

                # Yield only newly generated tokens
                diff_tokens = new_tokens[len(old_tokens) :]
                if diff_tokens:
                    yield "".join(diff_tokens)

                last_output = new_output  # Update last output

                if next_token == self.eos_id:
                    break
