"""
Copyright © 2023 Austin Berrio
Script: mini.engine.generator
Description: Simple completions for text-to-text generation with streaming output.
"""

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
        self.pad_id = config.pad_id
        self.bos_id = config.bos_id
        self.eos_id = config.eos_id
        self.max_seq_len = config.max_seq_len
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

    def load(self) -> None:
        """Load the model state and set to evaluation mode."""
        self.state.load(train=False)
        self.state.model.to(self.device)
        self.state.model.eval()

    def encode(self, prompt: str) -> List[int]:
        return self.processor.encode(prompt, add_bos=True, add_eos=False)

    def decode(self, encodings: Union[int, List[int]]) -> str:
        return self.processor.decode(encodings)

    def batch(self, encodings: List[int]) -> torch.Tensor:
        return torch.tensor([encodings], dtype=torch.long, device=self.device)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.state.model(x)[:, -1, :]

    def sample(self, x: torch.Tensor, past_tokens: List[int]) -> int:
        return self.sampler.sample(x, past_tokens=past_tokens)

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
