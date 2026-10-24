"""Versioned, lossless tensor bytes inside ordinary uncompressed tar members."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any, Iterable

import torch

MAGIC = b"LSRAW001"
DTYPES = {
    str(value): value
    for value in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.int64,
        torch.int32,
        torch.int8,
        torch.uint8,
        torch.bool,
    )
}


@dataclass
class RawPayload:
    header: bytes
    tensors: tuple[torch.Tensor, ...]

    @property
    def size(self) -> int:
        return len(self.header) + sum(
            value.numel() * value.element_size() for value in self.tensors
        )

    def chunks(self) -> Iterable[memoryview | bytes]:
        yield self.header
        for value in self.tensors:
            yield memoryview(value.reshape(-1).view(torch.uint8).numpy())


def encode_raw_sample(
    source_index: int,
    tokens: dict[str, torch.Tensor],
    aux: torch.Tensor,
    final: torch.Tensor,
    layer_ids: list[int],
    final_layer_id: int,
) -> RawPayload:
    """Store one packed aux slab; layer descriptors are strided views into it."""
    if aux.device.type != "cpu" or final.device.type != "cpu":
        raise ValueError("raw encoding requires completed CPU tensors")
    tensors = tuple(value.contiguous() for value in [*tokens.values(), aux, final])
    descriptors = {}
    offset = 0
    for key, value in zip([*tokens, "aux", "final"], tensors):
        descriptors[key] = {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "offset": offset,
            "bytes": value.numel() * value.element_size(),
        }
        offset += descriptors[key]["bytes"]
    descriptor = {
        "version": 1,
        "sample_index": source_index,
        "layer_ids": layer_ids,
        "final_layer_id": final_layer_id,
        "tensors": descriptors,
    }
    encoded = json.dumps(descriptor, separators=(",", ":")).encode()
    header = MAGIC + struct.pack("<Q", len(encoded)) + encoded
    header += b"\0" * (-len(header) % 64)
    return RawPayload(header, tensors)


def decode_raw_sample(data: bytes | bytearray) -> dict[str, Any]:
    """Validate the descriptor and return owning tensor views, without pickle."""
    storage = bytearray(data) if not isinstance(data, bytearray) else data
    if len(storage) < 16 or storage[:8] != MAGIC:
        raise ValueError("invalid tar-raw-v1 magic")
    length = struct.unpack("<Q", storage[8:16])[0]
    if length > 1024 * 1024 or 16 + length > len(storage):
        raise ValueError("invalid raw descriptor size")
    descriptor = json.loads(storage[16 : 16 + length])
    if descriptor["version"] != 1:
        raise ValueError("unsupported raw descriptor version")
    base = (16 + length + 63) // 64 * 64
    values = {}
    expected_offset = 0
    for key, item in descriptor["tensors"].items():
        dtype = DTYPES.get(item["dtype"])
        shape = item["shape"]
        if (
            dtype is None
            or not isinstance(shape, list)
            or any(not isinstance(v, int) or v < 0 for v in shape)
        ):
            raise ValueError("invalid raw dtype or shape")
        count = 1
        for dimension in shape:
            count *= dimension
        size = count * torch.empty((), dtype=dtype).element_size()
        if (
            item["offset"] != expected_offset
            or item["bytes"] != size
            or base + expected_offset + size > len(storage)
        ):
            raise ValueError("invalid raw tensor extent")
        values[key] = (
            torch.frombuffer(
                storage, dtype=dtype, count=count, offset=base + expected_offset
            ).reshape(shape)
            if count
            else torch.empty(shape, dtype=dtype)
        )
        expected_offset += size
    if base + expected_offset != len(storage):
        raise ValueError("unexpected raw trailing data")
    final = values.pop("final")
    aux = values.pop("aux")
    layers = descriptor["layer_ids"]
    if (
        len(set(layers)) != len(layers)
        or final.ndim != 2
        or aux.shape != (final.shape[0], len(layers) * final.shape[1])
    ):
        raise ValueError("invalid raw layer dimensions")
    if descriptor["final_layer_id"] in layers:
        raise ValueError("final normalized layer must be stored separately")
    hidden = {
        layer: aux[:, index * final.shape[1] : (index + 1) * final.shape[1]]
        for index, layer in enumerate(layers)
    }
    return {
        "schema_version": 2,
        "sample_index": descriptor["sample_index"],
        **values,
        "hidden_states": hidden,
        "final_hidden_state": final,
    }
