"""Fixed-capacity target KV storage for speculative decoding benchmarks.

Matches prealloc_kvcache_bench: allocate full storage on the first (prefill)
write, append in place, and expose only the valid prefix to attention. Allocation
is included in prefill, outside decode timing. Cropping never reallocates storage.
"""

import inspect

import torch
from transformers.cache_utils import Cache, DynamicCache, StaticLayer


class PreallocatedLayer(StaticLayer):
    # Storage addresses are static, but the visible prefix has a dynamic shape.
    is_compileable = False

    def __init__(self, max_cache_len: int):
        if max_cache_len < 1:
            raise ValueError("max_cache_len must be positive")
        super().__init__(max_cache_len=max_cache_len)
        self.logical_length = 0
        self._separate_value_initialization = (
            len(inspect.signature(self.lazy_initialization).parameters) > 1
        )

    def update(self, key_states, value_states, cache_kwargs=None):
        append_length = key_states.shape[-2]
        start = self.logical_length
        end = start + append_length
        if end > self.max_cache_len:
            raise ValueError(
                f"KV cache capacity exceeded: need {end}, max {self.max_cache_len}"
            )
        if not self.is_initialized:
            if self._separate_value_initialization:
                self.lazy_initialization(key_states, value_states)
            else:
                self.lazy_initialization(key_states)
        self.keys.narrow(2, start, append_length).copy_(key_states)
        self.values.narrow(2, start, append_length).copy_(value_states)
        self.logical_length = end
        return self.keys.narrow(2, 0, end), self.values.narrow(2, 0, end)

    def get_mask_sizes(self, query_length: int | torch.Tensor) -> tuple[int, int]:
        length = (
            query_length.shape[0]
            if isinstance(query_length, torch.Tensor)
            else query_length
        )
        return self.logical_length + int(length), 0

    def get_seq_length(self) -> int:
        return self.logical_length

    def crop(self, max_length: int):
        if max_length < 0:
            max_length = max(self.logical_length + max_length, 0)
        self.logical_length = min(self.logical_length, max_length)

    def reset(self):
        self.logical_length = 0


class PreallocatedTargetCache(Cache):
    def __init__(self, num_layers: int, max_cache_len: int):
        super().__init__(
            layers=[PreallocatedLayer(max_cache_len) for _ in range(num_layers)]
        )

    def storage_ptrs(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (layer.keys.data_ptr(), layer.values.data_ptr())
            for layer in self.layers
            if layer.is_initialized
        )


def make_target_cache(target, max_cache_len: int) -> Cache:
    config = getattr(target.config, "text_config", target.config)
    if any("linear" in layer for layer in getattr(config, "layer_types", [])):
        # Native hybrid cache layers own convolution/recurrent state and the
        # recording API used by DLite's rollback transactions.
        return DynamicCache(config=config)
    return PreallocatedTargetCache(int(config.num_hidden_layers), int(max_cache_len))
