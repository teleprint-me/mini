"""
Copyright © 2023 Austin Berrio
Script: mini.engine.generator
Description: Simple completions for text-to-text generation with streaming output.
"""

import functools
from typing import List, Optional, Union

import regex as re  # Use `regex`, not `re`
import torch
from sentencepiece import SentencePieceProcessor

from mini.config.generator import ConfigGenerator
from mini.engine.sampler import EngineSampler
from mini.engine.state import EngineState


class EngineGenerator:
    def __init__(self, config: ConfigGenerator):
        self.config = config
        self.device = config.device
        self.load()

    @property
    def processor(self) -> SentencePieceProcessor:
        return self.config.processor

    @property
    def state(self) -> EngineState:
        return self.config.state

    @property
    def sampler(self) -> EngineSampler:
        return self.config.sampler

    @functools.cached_property
    def pad_id(self) -> int:
        return max(self.processor.pad_id(), 0) if self.processor else 0

    @functools.cached_property
    def bos_id(self) -> int:
        return self.processor.bos_id() if self.processor else 0  # Safe fallback

    @functools.cached_property
    def eos_id(self) -> int:
        return self.processor.eos_id() if self.processor else 0  # Safe fallback

    @property
    def max_seq_len(self) -> int:
        return self.state.model.max_seq_len

    def load(self) -> None:
        """Load the model state and set to evaluation mode."""
        self.state.load(training_mode=False)
        self.state.model.to(self.device)
        self.state.model.eval()

    def encode(self, prompt: str) -> List[int]:
        return self.processor.encode(prompt, add_bos=True, add_eos=False)

    def decode(self, encodings: Union[int, List[int]]) -> str:
        return self.processor.decode(encodings)

    def batch(self, encodings: List[int]) -> torch.Tensor:
        """Converts tokenized input to a batch tensor and pads to max_seq_len."""
        seq_len = len(encodings)
        pad_length = max(0, self.max_seq_len - seq_len)  # Compute padding length

        # Pad input to `max_seq_len` with `pad_id`
        padded = encodings + [self.pad_id] * pad_length

        # Convert to tensor
        return torch.tensor([padded], dtype=torch.long, device=self.device)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.state.model(x)[:, -1, :]

    def sample(self, x: torch.Tensor, past_tokens: List[int]) -> int:
        return self.sampler.sample(x, past_tokens=past_tokens)

    def cat(self, x: torch.Tensor, token: int) -> torch.Tensor:
        """Replaces the first PAD token or appends if sequence is full."""
        pad_mask = (x == self.pad_id).nonzero(as_tuple=True)

        if pad_mask[1].numel() > 0:  # If there's a PAD token
            first_pad_idx = pad_mask[1][0].item()  # Get first PAD index
            x[:, first_pad_idx] = token  # Replace it with new token
        else:  # Context full, no padding left
            if x.shape[1] >= self.max_seq_len:
                x = torch.cat(
                    [x[:, 1:], torch.tensor([[token]], device=self.device)], dim=1
                )  # Shift left and append a new token
            else:
                x = torch.cat(
                    [x, torch.tensor([[token]], device=self.device)], dim=1
                )  # Append a new token

        return x

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
