import warnings
from typing import Callable, List

import torch
from transformers.generation import LogitsProcessor


class ConstrainedLogitsProcessor(LogitsProcessor):
    """Mask logits based on tokens generated after the real prompt boundary."""

    def __init__(self, prefix_allowed_tokens_fn: Callable[[int, list[int]], List[int]],
                 num_beams: int, prompt_lengths: int | list[int], eos_token_id: int | None = None):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams
        self.prompt_lengths = [prompt_lengths] if isinstance(prompt_lengths, int) else list(prompt_lengths)
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        mask = torch.full_like(scores, float("-inf"))
        for row, sent in enumerate(input_ids):
            batch_id = row // self._num_beams
            prompt_length = self.prompt_lengths[min(batch_id, len(self.prompt_lengths) - 1)]
            generated = sent[prompt_length:].tolist()
            allowed = self._prefix_allowed_tokens_fn(batch_id, generated)
            if not allowed:
                warnings.warn(f"No valid constrained-decoding continuation for {generated}")
                if self.eos_token_id is not None:
                    allowed = [self.eos_token_id]
            mask[row, allowed] = 0
        return scores + mask
