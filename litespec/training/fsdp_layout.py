"""Version-isolated FSDP1 layout adapter; no optimizer mathematics here."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    shape: tuple[int, ...]
    optimizer: str


@dataclass(frozen=True)
class ParameterLayout:
    spec: ParameterSpec
    # Parameter-relative flattened (offset, length), in process-group rank order.
    shards: tuple[tuple[int, int], ...]

    def __post_init__(self):
        cursor = 0
        for offset, length in sorted(s for s in self.shards if s[1]):
            if offset != cursor or length < 0:
                raise ValueError(f"Overlapping/incomplete shards for {self.spec.name}")
            cursor += length
        if any(offset < 0 or length < 0 for offset, length in self.shards):
            raise ValueError(f"Invalid shard for {self.spec.name}")
        if cursor != math.prod(self.spec.shape):
            raise ValueError(f"Shards do not cover {self.spec.name}: {self.shards}")

    def state_dict(self):
        return asdict(self)


def local_layouts(specs):
    return [ParameterLayout(s, ((0, math.prod(s.shape)),)) for s in specs]


def build_fsdp_layouts(fsdp_model, parameters, specs, process_group=None):
    """Read 2.9.1 metadata once, then validate a rank-consistent routing table.

    All private FSDP access is deliberately confined to this function. Parameters
    must have been registered before wrapping, including future-unfrozen ones.
    """
    rank = dist.get_rank(process_group) if dist.is_initialized() else 0
    world = dist.get_world_size(process_group) if dist.is_initialized() else 1
    error = None
    local = []
    try:
        if torch.__version__.split("+")[0] != "2.9.1":
            raise RuntimeError("Muon FSDP1 layout adapter requires torch==2.9.1")
        entries = {}
        for module in FSDP.fsdp_modules(fsdp_model):
            if not module._use_orig_params or module.cpu_offload.offload_params:
                raise ValueError("Muon requires use_orig_params=True, no CPU offload")
            if module.sharding_strategy not in (
                ShardingStrategy.SHARD_GRAD_OP,
                ShardingStrategy.NO_SHARD,
            ) or (world > 1 and module.sharding_strategy == ShardingStrategy.NO_SHARD):
                raise ValueError(
                    "Muon supports SHARD_GRAD_OP (NO_SHARD only on one rank)"
                )
            if dist.is_initialized() and (
                dist.get_process_group_ranks(module.process_group)
                != dist.get_process_group_ranks(process_group or dist.group.WORLD)
            ):
                raise ValueError("Muon and FSDP must use the same process group")
            handle = module._handle
            if handle is None:
                continue
            flat = handle.flat_param
            for p, shape, shard_info in zip(
                flat._params, flat._shapes, flat._shard_param_infos, strict=True
            ):
                shard = (
                    (int(shard_info.intra_param_start_idx), int(shard_info.numel_in_shard))
                    if shard_info.in_shard
                    else (0, 0)
                )
                entries[id(p)] = (tuple(shape), shard)
        if len(parameters) != len(specs):
            raise ValueError("Parameter/spec count mismatch")
        for p, spec in zip(parameters, specs, strict=True):
            if id(p) not in entries:
                raise ValueError(
                    f"Trainable parameter is not FSDP-managed: {spec.name}"
                )
            shape, shard = entries[id(p)]
            if shape != spec.shape or p.numel() != shard[1]:
                raise ValueError(f"Unexpected FSDP layout for {spec.name}")
            local.append(shard)
    except (AttributeError, ValueError, RuntimeError, TypeError) as exc:
        error = f"rank {rank}: {exc}"

    payload = (error, specs, local)
    gathered = [None] * world
    if world > 1:
        dist.all_gather_object(gathered, payload, group=process_group)
    else:
        gathered[0] = payload
    errors = [item[0] for item in gathered if item[0]]
    if errors:
        raise ValueError("Unsupported Muon FSDP layout: " + "; ".join(errors))
    if any(item[1] != specs for item in gathered):
        raise ValueError("Muon parameter registration differs between ranks")
    return [
        ParameterLayout(spec, tuple(item[2][i] for item in gathered))
        for i, spec in enumerate(specs)
    ]
