"""Reader and batching utilities for versioned ``regen_full`` tar caches."""

from __future__ import annotations

import hashlib
import json
import math
import random
import tarfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler

REGEN_FULL_SCHEMA_VERSION = 2
SAMPLE_INDEX_VERSION = 1


class HiddenCacheError(ValueError):
    """Raised when a regen_full cache cannot safely be used for training."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_model(expected: str, actual: str) -> bool:
    expected_path = Path(expected)
    actual_path = Path(actual)
    if expected_path.exists() and actual_path.exists():
        return expected_path.resolve() == actual_path.resolve()
    return expected.rstrip("/\\") == actual.rstrip("/\\")


def load_hidden_cache_manifest(
    root: str | Path,
    *,
    target_model: str,
    num_hidden_layers: int,
    hidden_size: int,
    required_layer_ids: set[int],
    max_length: int,
    verify_checksums: bool = True,
) -> dict[str, Any]:
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise HiddenCacheError(f"regen_full manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HiddenCacheError(f"cannot read regen_full manifest: {exc}") from exc

    if manifest.get("schema_version") != REGEN_FULL_SCHEMA_VERSION:
        raise HiddenCacheError(
            f"regen_full schema must be {REGEN_FULL_SCHEMA_VERSION}; regenerate the cache"
        )
    if manifest.get("save_mode") != "regen_full" or not manifest.get("complete"):
        raise HiddenCacheError("regen_full cache is not marked complete")
    if manifest.get("sample_index_version") != SAMPLE_INDEX_VERSION:
        raise HiddenCacheError(
            "regen_full cache has no lightweight sample index; regenerate the "
            "cache with the current prepare_hidden_states.py"
        )
    if not _same_model(target_model, str(manifest.get("target_model", ""))):
        raise HiddenCacheError(
            "regen_full target model does not match training target: "
            f"cache={manifest.get('target_model')!r}, training={target_model!r}"
        )
    if int(manifest.get("num_hidden_layers", -1)) != int(num_hidden_layers):
        raise HiddenCacheError("regen_full target layer count does not match training")
    if int(manifest.get("hidden_size", -1)) != int(hidden_size):
        raise HiddenCacheError("regen_full hidden size does not match training")
    if int(max_length) > int(manifest.get("max_length", -1)):
        raise HiddenCacheError(
            f"training max_length={max_length} exceeds cache max_length="
            f"{manifest.get('max_length')}"
        )

    final_layer_id = int(manifest.get("final_norm_layer_id", -1))
    if final_layer_id != int(num_hidden_layers) - 1:
        raise HiddenCacheError("regen_full final-norm layer ID is invalid")
    stored_layer_ids = {int(value) for value in manifest.get("stored_hidden_layer_ids", [])}
    missing_layer_ids = sorted(set(required_layer_ids) - stored_layer_ids)
    if final_layer_id not in stored_layer_ids:
        missing_layer_ids = sorted(set(missing_layer_ids) | {final_layer_id})
    if missing_layer_ids:
        needed_layer_ids = sorted(set(required_layer_ids) | {final_layer_id})
        raise HiddenCacheError(
            f"regen_full is missing required hidden layers {missing_layer_ids}; "
            f"regenerate with --target-layer-ids {','.join(map(str, needed_layer_ids))}"
        )

    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise HiddenCacheError("regen_full manifest contains no shards")
    expected_start = 0
    counted = 0
    for shard in shards:
        if not isinstance(shard, dict):
            raise HiddenCacheError("regen_full shard entry must be an object")
        path = root / str(shard.get("path", ""))
        if not path.is_file():
            raise HiddenCacheError(f"regen_full shard does not exist: {path}")
        start = int(shard.get("success_start", -1))
        end = int(shard.get("success_end", -1))
        count = int(shard.get("sample_count", -1))
        if start != expected_start or end != start + count - 1 or count <= 0:
            raise HiddenCacheError(f"regen_full shard range is invalid: {path}")
        source_indices = shard.get("source_indices", [])
        if len(source_indices) != count:
            raise HiddenCacheError(
                f"regen_full shard source index count is invalid: {path}"
            )
        sample_metadata = shard.get("sample_metadata", [])
        if len(sample_metadata) != count:
            raise HiddenCacheError(
                f"regen_full shard sample metadata count is invalid: {path}"
            )
        for value in sample_metadata:
            _validate_sample_metadata(value, path)
        if verify_checksums and _sha256(path) != shard.get("sha256"):
            raise HiddenCacheError(f"regen_full shard checksum mismatch: {path}")
        expected_start = end + 1
        counted += count
    if counted != int(manifest.get("successful_samples", -1)):
        raise HiddenCacheError("regen_full successful sample count is inconsistent")
    return manifest


def _validate_sample_metadata(value: Any, path: Path | None = None) -> None:
    location = f" in {path}" if path is not None else ""
    if not isinstance(value, dict):
        raise HiddenCacheError(
            f"regen_full sample metadata must be an object{location}"
        )
    length = value.get("sequence_length")
    count = value.get("supervised_token_count")
    runs = value.get("loss_mask_runs")
    if not isinstance(length, int) or length <= 0:
        raise HiddenCacheError(f"invalid indexed sequence length{location}")
    if not isinstance(count, int) or count < 0 or count > length:
        raise HiddenCacheError(f"invalid indexed supervision count{location}")
    if not isinstance(runs, list):
        raise HiddenCacheError(f"invalid indexed loss-mask runs{location}")
    previous_end = 0
    observed_count = 0
    for run in runs:
        if (
            not isinstance(run, list)
            or len(run) != 2
            or not all(isinstance(item, int) for item in run)
        ):
            raise HiddenCacheError(f"invalid indexed loss-mask run{location}")
        start, end = run
        if start < previous_end or start < 0 or end <= start or end > length:
            raise HiddenCacheError(f"invalid indexed loss-mask range{location}")
        observed_count += end - start
        previous_end = end
    if observed_count != count:
        raise HiddenCacheError(f"indexed supervision count mismatch{location}")


def sample_metadata_has_valid_anchor(
    metadata: dict[str, Any], *, block_size: int, max_length: int
) -> bool:
    """Evaluate anchor eligibility using only the lightweight cache index."""
    _validate_sample_metadata(metadata)
    block_size = int(block_size)
    effective_length = min(int(metadata["sequence_length"]), int(max_length))
    if effective_length < block_size:
        return False
    runs = metadata["loss_mask_runs"]
    supervised_count = sum(
        max(0, min(end, effective_length) - start)
        for start, end in runs
        if start < effective_length
    )
    if supervised_count < 2 * block_size:
        return False
    max_anchor = effective_length - block_size
    return any(
        start <= max_anchor and min(end, effective_length) - start >= 2
        for start, end in runs
        if start < effective_length
    )


class RegenFullDataset(Dataset):
    """Map-style dataset backed by regen_full tar members."""

    def __init__(
        self,
        root: str | Path,
        manifest: dict[str, Any],
        *,
        required_layer_ids: set[int],
        max_length: int,
        max_open_archives: int = 4,
    ) -> None:
        self.root = Path(root)
        self.manifest = manifest
        self.required_layer_ids = set(required_layer_ids)
        self.final_layer_id = int(manifest["final_norm_layer_id"])
        self.hidden_size = int(manifest["hidden_size"])
        self.max_length = int(max_length)
        self.max_open_archives = max(1, int(max_open_archives))
        self.entries: list[tuple[Path, str, int, dict[str, Any]]] = []
        for shard in manifest["shards"]:
            path = self.root / shard["path"]
            start = int(shard["success_start"])
            for offset, (source_index, sample_metadata) in enumerate(
                zip(shard["source_indices"], shard["sample_metadata"])
            ):
                ordinal = start + offset
                self.entries.append(
                    (
                        path,
                        f"samples/{ordinal:012d}.pt",
                        int(source_index),
                        sample_metadata,
                    )
                )
        self._archives: OrderedDict[Path, tarfile.TarFile] = OrderedDict()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_archives"] = OrderedDict()
        return state

    def close(self) -> None:
        archives = getattr(self, "_archives", {})
        for archive in archives.values():
            archive.close()
        archives.clear()

    def __del__(self):
        self.close()

    def __len__(self) -> int:
        return len(self.entries)

    def _load_payload(self, index: int) -> dict[str, Any]:
        path, member_name, source_index, _ = self.entries[index]
        archive = self._archives.get(path)
        if archive is None:
            if len(self._archives) >= self.max_open_archives:
                _, stale = self._archives.popitem(last=False)
                stale.close()
            # Reused per worker; the LRU bound prevents persistent workers from
            # eventually holding one descriptor for every shard.
            archive = tarfile.open(path, "r:*")  # noqa: SIM115
            self._archives[path] = archive
        else:
            self._archives.move_to_end(path)
        try:
            member = archive.getmember(member_name)
            handle = archive.extractfile(member)
            if handle is None:
                raise HiddenCacheError(f"cannot read {member_name} from {path}")
            payload = torch.load(handle, map_location="cpu", weights_only=True)
        except (KeyError, OSError, RuntimeError, tarfile.TarError) as exc:
            raise HiddenCacheError(
                f"cannot load {member_name} from {path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise HiddenCacheError(f"sample {member_name} is not an object")
        if payload.get("schema_version") != REGEN_FULL_SCHEMA_VERSION:
            raise HiddenCacheError(f"sample {member_name} has an unsupported schema")
        if int(payload.get("sample_index", -1)) != source_index:
            raise HiddenCacheError(f"sample index mismatch in {member_name}")
        return payload

    def eligible_indices(self, *, block_size: int) -> list[int]:
        """Return valid entry indices without opening a tar file."""
        return [
            index
            for index, (_, _, _, metadata) in enumerate(self.entries)
            if sample_metadata_has_valid_anchor(
                metadata, block_size=block_size, max_length=self.max_length
            )
        ]

    def shard_key(self, index: int) -> Path:
        return self.entries[index][0]

    def __getitem__(self, index: int) -> dict[str, Any]:
        payload = self._load_payload(index)
        input_ids = torch.as_tensor(payload.get("input_ids"))[: self.max_length]
        attention_mask = torch.as_tensor(payload.get("attention_mask"))[
            : self.max_length
        ]
        loss_mask = torch.as_tensor(payload.get("loss_mask"))[: self.max_length]
        length = input_ids.numel()
        if input_ids.ndim != 1 or length == 0:
            raise HiddenCacheError("cached input_ids must be a non-empty vector")
        if (
            attention_mask.shape != input_ids.shape
            or loss_mask.shape != input_ids.shape
        ):
            raise HiddenCacheError("cached token and mask shapes do not match")

        raw_hidden = payload.get("hidden_states")
        if not isinstance(raw_hidden, dict):
            raise HiddenCacheError("cached hidden_states must be a dictionary")
        hidden_states = {int(key): value for key, value in raw_hidden.items()}
        final_hidden = payload.get("final_hidden_state")
        if not isinstance(final_hidden, torch.Tensor):
            raise HiddenCacheError("cached final_hidden_state is missing")
        hidden_states[self.final_layer_id] = final_hidden
        selected = {}
        for layer_id in sorted(self.required_layer_ids | {self.final_layer_id}):
            value = hidden_states.get(layer_id)
            if not isinstance(value, torch.Tensor):
                raise HiddenCacheError(f"cached layer {layer_id} is missing")
            value = value[: self.max_length]
            if (
                value.ndim != 2
                or value.shape[0] != length
                or value.shape[1] != self.hidden_size
            ):
                raise HiddenCacheError(f"cached layer {layer_id} has invalid shape")
            if not torch.isfinite(value).all():
                raise HiddenCacheError(f"cached layer {layer_id} contains NaN or Inf")
            selected[layer_id] = value
        return {
            "sample_index": int(payload["sample_index"]),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "hidden_states": selected,
            "final_hidden_state": selected[self.final_layer_id],
            "final_norm_layer_id": self.final_layer_id,
        }


class ShardAwareDistributedSampler(Sampler[int]):
    """Shuffle by shard while giving each DP rank a mostly contiguous shard slice."""

    def __init__(
        self,
        dataset: RegenFullDataset,
        eligible_indices: list[int],
        *,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        if num_replicas <= 0 or rank < 0 or rank >= num_replicas:
            raise ValueError("invalid distributed sampler rank configuration")
        self.dataset = dataset
        self.eligible_indices = list(eligible_indices)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = math.ceil(len(self.eligible_indices) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _global_indices(self) -> list[int]:
        groups: dict[Path, list[int]] = {}
        for index in self.eligible_indices:
            groups.setdefault(self.dataset.shard_key(index), []).append(index)
        shard_keys = list(groups)
        if self.shuffle:
            generator = random.Random(self.seed + self.epoch)
            generator.shuffle(shard_keys)
            # Preserve member order inside each tar. This avoids random seeks (and
            # repeated decompression for .tar.gz) while still changing shard order
            # deterministically every epoch.
        values = [index for key in shard_keys for index in groups[key]]
        if values and len(values) < self.total_size:
            values += [
                values[index % len(values)]
                for index in range(self.total_size - len(values))
            ]
        return values

    def __iter__(self):
        values = self._global_indices()
        start = self.rank * self.num_samples
        return iter(values[start : start + self.num_samples])


class RegenFullCollator:
    def __init__(self, pad_token_id: int, pad_to_length: int | None = None) -> None:
        self.pad_token_id = int(pad_token_id)
        self.pad_to_length = pad_to_length

    @staticmethod
    def _pad_to(tensor: torch.Tensor, length: int) -> torch.Tensor:
        if tensor.shape[1] == length:
            return tensor
        shape = (tensor.shape[0], length - tensor.shape[1], *tensor.shape[2:])
        return torch.cat((tensor, tensor.new_zeros(shape)), dim=1)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise HiddenCacheError("cannot collate an empty regen_full batch")
        seq_len = max(item["input_ids"].numel() for item in features)
        if self.pad_to_length is not None:
            if seq_len > self.pad_to_length:
                raise HiddenCacheError("cached sequence exceeds fixed padding length")
            seq_len = self.pad_to_length

        def pad_vectors(key: str, padding_value: int) -> torch.Tensor:
            value = pad_sequence(
                [item[key] for item in features],
                batch_first=True,
                padding_value=padding_value,
            )
            return self._pad_to(value, seq_len)

        layer_ids = sorted(features[0]["hidden_states"])
        if any(sorted(item["hidden_states"]) != layer_ids for item in features):
            raise HiddenCacheError("cached samples contain different hidden layers")
        final_layer_id = int(features[0]["final_norm_layer_id"])
        if any(int(item["final_norm_layer_id"]) != final_layer_id for item in features):
            raise HiddenCacheError("cached samples disagree on final-norm layer ID")
        hidden_states = {}
        for layer_id in layer_ids:
            value = pad_sequence(
                [item["hidden_states"][layer_id] for item in features],
                batch_first=True,
                padding_value=0.0,
            )
            hidden_states[layer_id] = self._pad_to(value, seq_len)
        return {
            "sample_index": torch.tensor(
                [item["sample_index"] for item in features], dtype=torch.long
            ),
            "input_ids": pad_vectors("input_ids", self.pad_token_id),
            "attention_mask": pad_vectors("attention_mask", 0),
            "loss_mask": pad_vectors("loss_mask", 0),
            "hidden_states": hidden_states,
            "final_norm_layer_id": final_layer_id,
        }
