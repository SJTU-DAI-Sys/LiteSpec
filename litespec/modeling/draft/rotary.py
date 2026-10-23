"""Keep RoPE frequencies consistent between fresh and reloaded BF16 drafts."""

from __future__ import annotations

from typing import Callable

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding


class FP32RotaryEmbedding(Qwen3RotaryEmbedding):
    """Match Transformers' FP32 RoPE loading policy during ``model.to`` too."""

    def _apply(self, fn: Callable, recurse: bool = True):
        frequencies = {
            name: value
            for name, value in self._buffers.items()
            if value is not None and value.is_floating_point()
        }
        result = super()._apply(fn, recurse=recurse)
        for name, original in frequencies.items():
            converted = self._buffers[name]
            self._buffers[name] = (
                converted.float()
                if original.is_meta
                else original.to(device=converted.device, dtype=torch.float32)
            )
        return result
