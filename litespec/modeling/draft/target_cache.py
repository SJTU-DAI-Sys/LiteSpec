"""Native target-cache creation and recurrent-state verification transactions."""

from __future__ import annotations

import torch
from torch import nn
from transformers import DynamicCache
from transformers.cache_utils import Cache


def _new_target_cache(target: nn.Module) -> Cache:
    """Create the cache shape required by both legacy and hybrid targets."""
    try:
        return DynamicCache(config=target.config)
    except TypeError:
        return DynamicCache()


def _begin_recurrent_cache_transaction(
    cache: Cache,
) -> list[tuple[object, dict]] | None:
    """Record recurrent states that cannot be cropped after block verification."""
    snapshots: list[tuple[object, dict]] = []
    for layer in getattr(cache, "layers", ()):
        recurrent_states = getattr(layer, "recurrent_states", None)
        if isinstance(recurrent_states, torch.Tensor):
            tensors = {
                name: value.clone()
                for name in ("conv_states", "recurrent_states")
                if isinstance(value := getattr(layer, name, None), torch.Tensor)
            }
            snapshots.append(
                (
                    layer,
                    {
                        "tensor_attributes": tensors,
                        "has_previous_state": layer.has_previous_state,
                    },
                )
            )
            continue
        if not isinstance(recurrent_states, dict):
            continue
        initialized = getattr(layer, "is_recurrent_states_initialized", {})
        layer_snapshot = {
            key: value.clone()
            for key, value in recurrent_states.items()
            if value is not None and bool(initialized.get(key, True))
        }
        if layer_snapshot:
            snapshots.append((layer, layer_snapshot))
    if not snapshots:
        return None
    if hasattr(cache, "activate_past_recording"):
        cache.activate_past_recording()
    return snapshots


def _restore_recurrent_cache_transaction(
    snapshots: list[tuple[object, dict]],
) -> None:
    for layer, layer_snapshot in snapshots:
        if "tensor_attributes" in layer_snapshot:
            for name, saved in layer_snapshot["tensor_attributes"].items():
                getattr(layer, name).copy_(saved)
            layer.has_previous_state = layer_snapshot["has_previous_state"]
            continue
        recurrent_states = layer.recurrent_states
        for key, saved in layer_snapshot.items():
            current = recurrent_states.get(key)
            if current is None:
                recurrent_states[key] = saved
            else:
                current.copy_(saved)


def _crop_target_cache(
    cache: Cache,
    *,
    current_length: int,
    tokens_to_remove: int,
) -> None:
    """Bridge absolute-size crop (Transformers 4.x) and relative crop (5.x)."""
    tokens_to_remove = int(tokens_to_remove)
    if tokens_to_remove < 0:
        raise ValueError("tokens_to_remove must be non-negative")
    if hasattr(cache, "is_croppable"):
        cache.crop(-tokens_to_remove)
    else:
        cache.crop(int(current_length) - tokens_to_remove)
