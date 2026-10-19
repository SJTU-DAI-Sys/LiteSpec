#!/usr/bin/env python3
"""Generate versioned DLite token/hidden-state tar shards.

Each final shard contains ``--shard-size`` successful samples (512 by default).
Failed samples are logged and do not consume shard capacity.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA_VERSION = 2
SAMPLE_INDEX_VERSION = 1
METADATA_MEMBER = "_metadata.json"


class HiddenStateError(RuntimeError):
    pass


def parse_layer_ids(value: str | None) -> list[int] | None:
    if value is None:
        return None
    try:
        result = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "layer IDs must be comma-separated integers"
        ) from exc
    if (
        not result
        or len(set(result)) != len(result)
        or any(item < 0 for item in result)
    ):
        raise argparse.ArgumentTypeError(
            "layer IDs must be unique non-negative integers"
        )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    model = parser.add_argument_group("model")
    model.add_argument("--target-model-path", required=True)
    model.add_argument("--target-model-backend", choices=("hf", "sglang"), default="hf")
    model.add_argument("--target-layer-ids", type=parse_layer_ids)
    model.add_argument("--trust-remote-code", action="store_true")
    model.add_argument("--model-download-dir")

    data = parser.add_argument_group("data")
    data.add_argument("--data-path", required=True)
    data.add_argument(
        "--output-path",
        help=(
            "Full-cache output directory. Defaults to "
            "./cache/data/regen_full/<input>_<model>."
        ),
    )
    data.add_argument("--chat-template", default="qwen")
    data.add_argument("--is-preformatted", action="store_true")
    data.add_argument("--max-length", type=int, default=4096)
    data.add_argument("--num-samples", type=int)
    data.add_argument("--build-dataset-num-proc", type=int, default=8)
    data.add_argument("--cache-dir", default="./cache/hidden_states")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--tp-size", type=int, default=1)
    runtime.add_argument("--batch-size", type=int, default=1)
    runtime.add_argument("--dist-timeout", type=int, default=2000)
    runtime.add_argument("--max-retries", type=int, default=2)
    runtime.add_argument("--fail-on-error", action="store_true")
    runtime.add_argument("--resume", action="store_true")
    runtime.add_argument("--retry-failures", action="store_true")
    runtime.add_argument("--shard-size", type=int, default=512)
    runtime.add_argument("--compress", action="store_true")

    sglang = parser.add_argument_group("sglang backend")
    sglang.add_argument("--sglang-attention-backend", default="flashinfer")
    sglang.add_argument("--sglang-mem-fraction-static", type=float, default=0.4)
    sglang.add_argument("--sglang-context-length", type=int)
    sglang.add_argument("--sglang-enable-nccl-nvls", action="store_true")
    sglang.add_argument("--sglang-enable-symm-mem", action="store_true")
    sglang.add_argument("--sglang-enable-torch-compile", action="store_true")
    sglang.add_argument("--sglang-enable-dp-attention", action="store_true")
    sglang.add_argument("--sglang-enable-dp-lm-head", action="store_true")
    sglang.add_argument("--sglang-enable-piecewise-cuda-graph", action="store_true")
    sglang.add_argument(
        "--sglang-piecewise-cuda-graph-max-tokens", type=int, default=4096
    )
    sglang.add_argument("--sglang-piecewise-cuda-graph-tokens", type=int, nargs="+")
    sglang.add_argument("--sglang-ep-size", type=int, default=1)
    sglang.add_argument("--sglang-max-running-requests", type=int)
    sglang.add_argument("--sglang-max-total-tokens", type=int)
    args = parser.parse_args(argv)
    for name in (
        "max_length",
        "batch_size",
        "tp_size",
        "shard_size",
        "build_dataset_num_proc",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_samples is not None and args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.max_retries < 0:
        parser.error("--max-retries cannot be negative")
    if args.retry_failures and not args.resume:
        parser.error("--retry-failures requires --resume")
    return args


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def tar_mode(compress: bool, read: bool = False) -> str:
    if read:
        return "r:*"
    return "w:gz" if compress else "w"


def add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    member_info = tarfile.TarInfo(name=name)
    member_info.size = len(data)
    member_info.mtime = 0
    archive.addfile(member_info, io.BytesIO(data))


def read_shard_metadata(path: Path) -> dict[str, Any]:
    try:
        with tarfile.open(path, "r:*") as archive:
            member = archive.getmember(METADATA_MEMBER)
            handle = archive.extractfile(member)
            if handle is None:
                raise HiddenStateError("metadata member is unreadable")
            metadata = json.loads(handle.read().decode("utf-8"))
            sample_members = [
                item
                for item in archive.getmembers()
                if item.isfile() and item.name.startswith("samples/")
            ]
    except Exception as exc:
        raise HiddenStateError(f"invalid shard {path}: {exc}") from exc
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise HiddenStateError(f"unsupported schema in {path}")
    if metadata.get("sample_count") != len(sample_members):
        raise HiddenStateError(f"sample count mismatch in {path}")
    if len(metadata.get("source_indices", [])) != len(sample_members):
        raise HiddenStateError(f"source index count mismatch in {path}")
    sample_metadata = metadata.get("sample_metadata")
    if sample_metadata is not None and len(sample_metadata) != len(sample_members):
        raise HiddenStateError(f"sample metadata count mismatch in {path}")
    return metadata


@dataclass
class CommittedShard:
    path: Path
    metadata: dict[str, Any]


class TarShardWriter:
    def __init__(
        self,
        directory: Path,
        prefix: str,
        sequence: int,
        compress: bool,
        kind: str,
        input_identity: dict[str, Any],
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        extension = ".tar.gz" if compress else ".tar"
        self.final_path = directory / f"{prefix}_{sequence:08d}{extension}"
        self.temp_path = directory / f".{self.final_path.name}.tmp"
        self.archive = tarfile.open(self.temp_path, tar_mode(compress))
        self.kind = kind
        self.sequence = sequence
        self.input_identity = input_identity
        self.source_indices: list[int] = []
        self.sample_metadata: list[dict[str, Any] | None] = []
        self.failures: list[dict[str, Any]] = []
        self.consumed_start: int | None = None
        self.consumed_end: int | None = None

    @property
    def sample_count(self) -> int:
        return len(self.source_indices)

    def mark_consumed(self, source_index: int) -> None:
        if self.consumed_start is None:
            self.consumed_start = source_index
        self.consumed_end = source_index

    def add_sample(
        self,
        source_index: int,
        payload: bytes,
        ordinal: int | None = None,
        sample_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.mark_consumed(source_index)
        member_index = source_index if ordinal is None else ordinal
        add_bytes(self.archive, f"samples/{member_index:012d}.pt", payload)
        self.source_indices.append(source_index)
        self.sample_metadata.append(sample_metadata)

    def add_failure(self, failure: dict[str, Any]) -> None:
        source_index = int(failure["source_index"])
        self.mark_consumed(source_index)
        self.failures.append(failure)

    def abort(self) -> None:
        try:
            self.archive.close()
        finally:
            self.temp_path.unlink(missing_ok=True)

    def commit(
        self,
        extra_metadata: dict[str, Any] | None = None,
        final_filename: str | None = None,
    ) -> CommittedShard:
        if self.consumed_start is None:
            self.abort()
            raise HiddenStateError("cannot commit an empty, non-progressing shard")
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "sequence": self.sequence,
            "sample_count": self.sample_count,
            "source_indices": self.source_indices,
            "sample_metadata": self.sample_metadata,
            "consumed_start": self.consumed_start,
            "consumed_end": self.consumed_end,
            "failures": self.failures,
            "input_identity": self.input_identity,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        add_bytes(
            self.archive,
            METADATA_MEMBER,
            json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode("utf-8"),
        )
        self.archive.close()
        checked = read_shard_metadata(self.temp_path)
        if checked["sample_count"] != self.sample_count:
            self.temp_path.unlink(missing_ok=True)
            raise HiddenStateError("temporary shard validation failed")
        final_path = (
            self.final_path.with_name(final_filename)
            if final_filename is not None
            else self.final_path
        )
        os.replace(self.temp_path, final_path)
        return CommittedShard(final_path, metadata)


def iter_shard_samples(
    path: Path,
) -> Iterator[tuple[int, bytes, dict[str, Any] | None]]:
    metadata = read_shard_metadata(path)
    with tarfile.open(path, "r:*") as archive:
        members = sorted(
            (
                item
                for item in archive.getmembers()
                if item.isfile() and item.name.startswith("samples/")
            ),
            key=lambda item: item.name,
        )
        if len(members) != len(metadata["source_indices"]):
            raise HiddenStateError(f"member/index mismatch in {path}")
        sample_metadata = metadata.get("sample_metadata") or [None] * len(members)
        if len(sample_metadata) != len(members):
            raise HiddenStateError(f"sample metadata count mismatch in {path}")
        for source_index, member, index_metadata in zip(
            metadata["source_indices"], members, sample_metadata
        ):
            handle = archive.extractfile(member)
            if handle is None:
                raise HiddenStateError(f"cannot read {member.name} in {path}")
            yield int(source_index), handle.read(), index_metadata


def discover_staging(
    output_root: Path, input_identity: dict[str, Any], rank: int | None = None
) -> list[CommittedShard]:
    pattern = f"rank_{rank:05d}/stage_*" if rank is not None else "rank_*/stage_*"
    shards = []
    for path in sorted((output_root / "staging").glob(pattern)):
        if path.name.startswith(".") or path.suffix not in {".tar", ".gz"}:
            continue
        metadata = read_shard_metadata(path)
        if metadata.get("kind") != "staging":
            continue
        if metadata.get("input_identity") != input_identity:
            raise HiddenStateError(f"staging shard belongs to another input: {path}")
        shards.append(CommittedShard(path, metadata))
    return shards


def repair_staging(
    output_root: Path,
    input_identity: dict[str, Any],
    rank: int,
    expected_start: int | None,
) -> list[CommittedShard]:
    directory = output_root / "staging" / f"rank_{rank:05d}"
    paths = sorted(
        path for path in directory.glob("stage_*") if path.suffix in {".tar", ".gz"}
    )
    valid: list[CommittedShard] = []
    next_source = expected_start
    expected_sequence = None
    truncate_at: int | None = None
    for index, path in enumerate(paths):
        try:
            metadata = read_shard_metadata(path)
            if metadata.get("input_identity") != input_identity:
                raise HiddenStateError(
                    f"staging shard belongs to another input: {path}"
                )
            if expected_sequence is None:
                expected_sequence = int(metadata.get("sequence", -1))
            if int(metadata.get("sequence", -1)) != expected_sequence:
                raise HiddenStateError(f"non-contiguous staging sequence at {path}")
            if next_source is None:
                next_source = int(metadata.get("consumed_start", -1))
            if int(metadata.get("consumed_start", -1)) != next_source:
                raise HiddenStateError(f"non-contiguous source range at {path}")
            next_source = int(metadata["consumed_end"]) + 1
            expected_sequence += 1
            valid.append(CommittedShard(path, metadata))
        except HiddenStateError as exc:
            if "another input" in str(exc):
                raise
            truncate_at = index
            break
    if truncate_at is not None:
        for path in paths[truncate_at:]:
            path.unlink(missing_ok=True)
    return valid


def finalized_rank_cursor(
    output_root: Path,
    input_identity: dict[str, Any],
    shard_size: int,
    rank: int,
    start: int,
    end: int,
) -> int:
    processed = {
        int(source_index)
        for shard in valid_final_shards(output_root, shard_size, input_identity)
        for source_index in shard.metadata["source_indices"]
        if start <= int(source_index) < end
    }
    error_path = output_root / "errors" / f"rank_{rank:05d}.jsonl"
    if error_path.exists():
        with error_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    source_index = int(json.loads(line)["source_index"])
                    if start <= source_index < end:
                        processed.add(source_index)
    cursor = start
    while cursor in processed:
        cursor += 1
    return cursor


def write_rank_errors(
    output_root: Path, rank: int, shards: list[CommittedShard]
) -> None:
    path = output_root / "errors" / f"rank_{rank:05d}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for shard in shards:
            for failure in shard.metadata.get("failures", []):
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def load_rank_errors(output_root: Path) -> list[dict[str, Any]]:
    failures = []
    for path in sorted((output_root / "errors").glob("rank_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    failures.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise HiddenStateError(
                        f"invalid rank error log {path}:{line_number}: {exc}"
                    ) from exc
    return failures


def valid_final_shards(
    output_root: Path, shard_size: int, input_identity: dict[str, Any]
) -> list[CommittedShard]:
    result = []
    expected_start = 0
    for path in sorted((output_root / "shards").glob("shard_*")):
        if path.name.startswith(".") or path.suffix not in {".tar", ".gz"}:
            continue
        try:
            metadata = read_shard_metadata(path)
            checksum_path = path.with_name(f"{path.name}.sha256")
            expected_checksum = checksum_path.read_text(encoding="utf-8").strip()
            if sha256_file(path) != expected_checksum:
                break
            if metadata.get("kind") != "final":
                break
            if metadata.get("input_identity") != input_identity:
                break
            if metadata.get("success_start") != expected_start:
                break
            if metadata["sample_count"] != shard_size:
                break
        except Exception:
            break
        result.append(CommittedShard(path, metadata))
        expected_start += metadata["sample_count"]
    return result


def finalize_staging(
    output_root: Path,
    input_identity: dict[str, Any],
    shard_size: int,
    compress: bool,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    staging = discover_staging(output_root, input_identity)
    staging.sort(
        key=lambda item: (int(item.metadata["consumed_start"]), item.path.name)
    )
    final_shards = valid_final_shards(output_root, shard_size, input_identity)
    if run_metadata.get("sample_index_version") == SAMPLE_INDEX_VERSION:
        for item in final_shards:
            metadata = item.metadata.get("sample_metadata")
            if not isinstance(metadata, list) or any(
                value is None for value in metadata
            ):
                raise HiddenStateError(
                    f"final shard lacks lightweight sample metadata: {item.path}; "
                    "regenerate the cache"
                )
    consumed_source_indices = {
        int(source_index)
        for item in final_shards
        for source_index in item.metadata["source_indices"]
    }
    success_ordinal = sum(item.metadata["sample_count"] for item in final_shards)
    final_sequence = len(final_shards)
    writer: TarShardWriter | None = None
    pending_stage_deletions: list[Path] = []

    def delete_committed_staging() -> None:
        while pending_stage_deletions:
            pending_stage_deletions.pop(0).unlink(missing_ok=True)

    for stage in staging:
        for source_index, payload, sample_metadata in iter_shard_samples(stage.path):
            if source_index in consumed_source_indices:
                continue
            if (
                run_metadata.get("sample_index_version") == SAMPLE_INDEX_VERSION
                and sample_metadata is None
            ):
                raise HiddenStateError(
                    f"staging shard lacks lightweight sample metadata: {stage.path}; "
                    "regenerate the cache"
                )
            if writer is None:
                writer = TarShardWriter(
                    output_root / "shards",
                    "shard_build",
                    final_sequence,
                    compress,
                    "final",
                    input_identity,
                )
            writer.add_sample(
                source_index,
                payload,
                ordinal=success_ordinal,
                sample_metadata=sample_metadata,
            )
            success_ordinal += 1
            if writer.sample_count == shard_size:
                success_start = success_ordinal - shard_size
                success_end = success_ordinal - 1
                extension = ".tar.gz" if compress else ".tar"
                committed = writer.commit(
                    {
                        "success_start": success_start,
                        "success_end": success_end,
                    },
                    final_filename=(
                        f"shard_{success_start:08d}_{success_end:08d}{extension}"
                    ),
                )
                checksum = sha256_file(committed.path)
                atomic_write_text(
                    committed.path.with_name(f"{committed.path.name}.sha256"),
                    checksum + "\n",
                )
                final_shards.append(committed)
                final_sequence += 1
                writer = None
                delete_committed_staging()
        pending_stage_deletions.append(stage.path)
        if writer is None:
            delete_committed_staging()

    if writer is not None:
        count = writer.sample_count
        success_start = success_ordinal - count
        success_end = success_ordinal - 1
        extension = ".tar.gz" if compress else ".tar"
        committed = writer.commit(
            {
                "success_start": success_start,
                "success_end": success_end,
                "final_partial": True,
            },
            final_filename=f"shard_{success_start:08d}_{success_end:08d}{extension}",
        )
        checksum = sha256_file(committed.path)
        atomic_write_text(
            committed.path.with_name(f"{committed.path.name}.sha256"),
            checksum + "\n",
        )
        final_shards.append(committed)
        delete_committed_staging()

    failures = load_rank_errors(output_root)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "input_identity": input_identity,
        **run_metadata,
        "shard_size": shard_size,
        "successful_samples": success_ordinal,
        "failed_samples": len(failures),
        "shards": [
            {
                "path": str(item.path.relative_to(output_root)),
                "sample_count": item.metadata["sample_count"],
                "source_indices": item.metadata["source_indices"],
                "sample_metadata": item.metadata["sample_metadata"],
                "success_start": item.metadata["success_start"],
                "success_end": item.metadata["success_end"],
                "sha256": sha256_file(item.path),
            }
            for item in final_shards
        ],
    }
    atomic_write_json(output_root / "manifest.json", manifest)

    combined_errors = output_root / "errors.jsonl"
    temporary_errors = output_root / ".errors.jsonl.tmp"
    with temporary_errors.open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
    os.replace(temporary_errors, combined_errors)

    # Final artifacts are durable at this point; remove empty rank directories
    # and any staging file left by an interrupted cleanup.
    shutil.rmtree(output_root / "staging", ignore_errors=True)
    return manifest


def normalize_hidden_states(
    hidden_states: Any, layer_ids: list[int] | None
) -> dict[int, Any]:
    if isinstance(hidden_states, dict):
        result = {int(key): value for key, value in hidden_states.items()}
    elif isinstance(hidden_states, (tuple, list)):
        keys = layer_ids if layer_ids is not None else list(range(len(hidden_states)))
        if len(keys) != len(hidden_states):
            raise HiddenStateError("hidden-state count does not match target layer IDs")
        result = dict(zip(keys, hidden_states))
    else:
        raise HiddenStateError("target returned an unsupported hidden-state container")
    if not result:
        raise HiddenStateError("target returned no hidden states")
    return result


def sample_payload(
    output: Any,
    batch_index: int,
    source_index: int,
    stored_layer_ids: list[int],
    final_norm_layer_id: int,
):
    import torch

    attention = output.attention_mask[batch_index]
    length = int(attention.sum().item())
    if length <= 0:
        raise HiddenStateError("sample has no attended tokens")
    target_hidden_states = normalize_hidden_states(output.hidden_states, stored_layer_ids)
    missing_layer_ids = sorted(set(stored_layer_ids) - set(target_hidden_states))
    if missing_layer_ids:
        raise HiddenStateError(f"target did not return requested layers {missing_layer_ids}")
    if final_norm_layer_id not in target_hidden_states:
        raise HiddenStateError(
            f"target did not return final-norm layer {final_norm_layer_id}"
        )
    sample_hidden = {}
    for layer_id, tensor in target_hidden_states.items():
        value = tensor[batch_index, :length].detach().cpu().contiguous()
        if not torch.isfinite(value).all():
            raise HiddenStateError(f"layer {layer_id} contains NaN or Inf")
        if value.ndim != 2 or value.shape[0] != length:
            raise HiddenStateError(
                f"layer {layer_id} has invalid shape {tuple(value.shape)}"
            )
        sample_hidden[layer_id] = value
    final_hidden_state = sample_hidden.pop(final_norm_layer_id)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "sample_index": source_index,
        "input_ids": output.input_ids[batch_index, :length].detach().cpu().contiguous(),
        "attention_mask": attention[:length].detach().cpu().contiguous(),
        "loss_mask": output.loss_mask[batch_index, :length].detach().cpu().contiguous(),
        "hidden_states": sample_hidden,
        "final_hidden_state": final_hidden_state,
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def sample_index_metadata(output: Any, batch_index: int) -> dict[str, Any]:
    """Build an exact, compact loss-mask index without serializing sample tensors."""
    import torch

    attention = output.attention_mask[batch_index].reshape(-1)
    length = int(attention.sum().item())
    if length <= 0:
        raise HiddenStateError("sample has no attended tokens")
    active = output.loss_mask[batch_index, :length].detach().reshape(-1).gt(0.5).cpu()
    if active.numel() != length:
        raise HiddenStateError("loss mask length does not match attended tokens")
    padded = torch.cat(
        (
            torch.zeros(1, dtype=torch.int8),
            active.to(torch.int8),
            torch.zeros(1, dtype=torch.int8),
        )
    )
    transitions = padded[1:] - padded[:-1]
    starts = torch.nonzero(transitions == 1, as_tuple=False).reshape(-1).tolist()
    ends = torch.nonzero(transitions == -1, as_tuple=False).reshape(-1).tolist()
    return {
        "sequence_length": length,
        "supervised_token_count": int(active.sum().item()),
        "loss_mask_runs": [[int(start), int(end)] for start, end in zip(starts, ends)],
    }


def collate_records(records: list[dict[str, Any]], pad_token_id: int):
    from torch.nn.utils.rnn import pad_sequence

    def values(key, padding):
        tensors = [item[key].reshape(-1) for item in records]
        return pad_sequence(tensors, batch_first=True, padding_value=padding)

    return {
        "input_ids": values("input_ids", pad_token_id),
        "attention_mask": values("attention_mask", 0),
        "loss_mask": values("loss_mask", 0),
    }


def failure(source_index: int, stage: str, exc: Exception) -> dict[str, Any]:
    return {
        "source_index": source_index,
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def generate_with_retries(model, batch, max_retries: int):
    last_error = None
    for _ in range(max_retries + 1):
        try:
            return model.generate_training_data(**batch, return_logits=False)
        except Exception as exc:
            last_error = exc
    raise last_error


def contiguous_range(total: int, rank: int, size: int) -> tuple[int, int]:
    base, remainder = divmod(total, size)
    start = rank * base + min(rank, remainder)
    end = start + base + (1 if rank < remainder else 0)
    return start, end


def _path_component(value: str) -> str:
    """Make a model or dataset name safe to use as one path component."""
    safe_component = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )
    return safe_component.strip("._-") or "data"


def default_output_path(data_path: str, target_model_path: str) -> Path:
    dataset = _path_component(Path(data_path).stem)
    model = _path_component(target_model_path.rstrip("/\\").split("/")[-1])
    return Path("./cache/data/regen_full") / f"{dataset}_{model}"


def run_generation(args: argparse.Namespace) -> dict[str, Any] | None:
    import torch
    import torch.distributed as dist
    from transformers import AutoConfig, AutoTokenizer

    from datasets import load_dataset
    from litespec.args import SGLangBackendArgs
    from litespec.data import build_training_dataset
    from litespec.distributed import (
        destroy_distributed,
        get_dp_group,
        get_tp_group,
        init_distributed,
        is_tp_rank_0,
    )
    from litespec.modeling.target import get_training_target_model

    input_path = Path(args.data_path)
    if not input_path.is_file():
        raise HiddenStateError(f"data path does not exist: {input_path}")
    output_root = (
        Path(args.output_path)
        if args.output_path
        else default_output_path(args.data_path, args.target_model_path)
    )
    identity = input_fingerprint(input_path)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = output_root / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != SCHEMA_VERSION:
                raise HiddenStateError(
                    f"existing cache schema is not {SCHEMA_VERSION}; "
                    "regenerate the cache"
                )
            if manifest.get("input_identity") != identity:
                raise HiddenStateError(
                    f"existing manifest belongs to another input: {manifest_path}"
                )
            if manifest.get("sample_index_version") != SAMPLE_INDEX_VERSION:
                raise HiddenStateError(
                    "existing cache has no lightweight sample index; regenerate it"
                )
            if args.resume and manifest.get("complete") and not args.retry_failures:
                if dist.get_rank() == 0:
                    print(
                        f"manifest already complete: {manifest_path} "
                        f"successful={manifest.get('successful_samples')} "
                        f"failed={manifest.get('failed_samples')}"
                    )
                return manifest if dist.get_rank() == 0 else None
        if not args.resume and any(output_root.iterdir()):
            raise HiddenStateError(
                f"output directory is not empty; use --resume: {output_root}"
            )
        if args.retry_failures:
            if dist.get_rank() == 0:
                for path in (
                    output_root / "staging",
                    output_root / "shards",
                    output_root / "errors",
                ):
                    shutil.rmtree(path, ignore_errors=True)
                for path in (manifest_path, output_root / "errors.jsonl"):
                    path.unlink(missing_ok=True)
            dist.barrier()
        for temp in output_root.glob("**/.*.tmp"):
            temp.unlink(missing_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(
            args.target_model_path,
            cache_dir=args.model_download_dir,
            trust_remote_code=args.trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = (
                tokenizer.eos_token_id or tokenizer.unk_token_id or 0
            )
        config = AutoConfig.from_pretrained(
            args.target_model_path,
            cache_dir=args.model_download_dir,
            trust_remote_code=args.trust_remote_code,
        )
        target_kwargs = {"trust_remote_code": args.trust_remote_code}
        if args.target_model_backend == "sglang":
            target_kwargs.update(SGLangBackendArgs.from_args(args).to_kwargs())
        target = get_training_target_model(
            args.target_model_path,
            backend=args.target_model_backend,
            torch_dtype=getattr(config, "dtype", getattr(config, "torch_dtype", None)),
            device="cuda" if args.target_model_backend == "hf" else None,
            cache_dir=args.model_download_dir,
            **target_kwargs,
        )
        num_hidden_layers = int(config.num_hidden_layers)
        hidden_size = int(config.hidden_size)
        final_norm_layer_id = num_hidden_layers - 1
        requested_layer_ids = (
            list(range(num_hidden_layers))
            if args.target_layer_ids is None
            else list(args.target_layer_ids)
        )
        invalid_layer_ids = [
            layer_id
            for layer_id in requested_layer_ids
            if not 0 <= layer_id < num_hidden_layers
        ]
        if invalid_layer_ids:
            raise HiddenStateError(
                f"target layer IDs {invalid_layer_ids} are outside [0, "
                f"{num_hidden_layers - 1}]"
            )
        stored_layer_ids = sorted(set(requested_layer_ids) | {final_norm_layer_id})
        target.set_capture_layers(stored_layer_ids)

        raw_dataset = load_dataset("json", data_files=str(input_path))["train"]
        if args.num_samples is not None:
            raw_dataset = raw_dataset.select(range(min(args.num_samples, len(raw_dataset))))
        cache_key = hashlib.sha256(
            json.dumps(
                {
                    "identity": identity,
                    "max_length": args.max_length,
                    "chat_template": args.chat_template,
                    "model": args.target_model_path,
                    "preformatted": args.is_preformatted,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

        def build_processed_dataset():
            return build_training_dataset(
                dataset=raw_dataset,
                tokenizer=tokenizer,
                chat_template=args.chat_template,
                max_length=args.max_length,
                shuffle_seed=None,
                num_proc=args.build_dataset_num_proc,
                cache_dir=str(Path(args.cache_dir) / "processed_dataset"),
                cache_key=cache_key,
                is_preformatted=args.is_preformatted,
                capture_errors=True,
            )

        processed = build_processed_dataset() if dist.get_rank() == 0 else None
        dist.barrier()
        if processed is None:
            processed = build_processed_dataset()

        dp_rank = dist.get_rank(get_dp_group())
        dp_size = dist.get_world_size(get_dp_group())
        start, end = contiguous_range(len(processed), dp_rank, dp_size)
        if is_tp_rank_0():
            rank_staging_dir = output_root / "staging" / f"rank_{dp_rank:05d}"
            has_staging = any(rank_staging_dir.glob("stage_*"))
            fallback_cursor = finalized_rank_cursor(
                output_root,
                identity,
                args.shard_size,
                dp_rank,
                start,
                end,
            )
            existing = repair_staging(
                output_root,
                identity,
                dp_rank,
                None if has_staging else fallback_cursor,
            )
        else:
            existing = []
            fallback_cursor = start
        resume_info = [
            (
                (
                    max(int(item.metadata["consumed_end"]) for item in existing) + 1
                    if existing
                    else fallback_cursor
                ),
                max((int(item.metadata["sequence"]) for item in existing), default=-1)
                + 1,
            )
        ]
        dist.broadcast_object_list(
            resume_info,
            src=dist.get_process_group_ranks(get_tp_group())[0],
            group=get_tp_group(),
        )
        cursor, sequence = resume_info[0]
        cursor = max(cursor, start)
        writer = (
            TarShardWriter(
                output_root / "staging" / f"rank_{dp_rank:05d}",
                "stage",
                sequence,
                args.compress,
                "staging",
                identity,
            )
            if is_tp_rank_0() and cursor < end
            else None
        )
        shard_successes = 0
        pending_records: list[dict[str, Any]] = []

        def record_error(error_value: dict[str, Any]) -> None:
            nonlocal writer
            if args.fail_on_error:
                raise HiddenStateError(json.dumps(error_value, ensure_ascii=False))
            if is_tp_rank_0():
                writer.add_failure(error_value)

        def commit_writer() -> None:
            nonlocal writer, sequence, shard_successes
            if is_tp_rank_0():
                writer.commit({"rank": dp_rank})
                sequence += 1
                writer = TarShardWriter(
                    output_root / "staging" / f"rank_{dp_rank:05d}",
                    "stage",
                    sequence,
                    args.compress,
                    "staging",
                    identity,
                )
            shard_successes = 0

        def handle_batch(records: list[dict[str, Any]]) -> None:
            nonlocal shard_successes
            if not records:
                return
            cpu_batch = collate_records(records, tokenizer.pad_token_id)
            gpu_batch = {
                key: value.cuda(non_blocking=True) for key, value in cpu_batch.items()
            }
            try:
                try:
                    batch_output = generate_with_retries(
                        target, gpu_batch, args.max_retries
                    )
                    outcomes = [
                        (batch_output, index, record, None)
                        for index, record in enumerate(records)
                    ]
                except Exception:
                    outcomes = []
                    for record in records:
                        single_cpu = collate_records([record], tokenizer.pad_token_id)
                        single_gpu = {
                            key: value.cuda(non_blocking=True)
                            for key, value in single_cpu.items()
                        }
                        try:
                            value = generate_with_retries(
                                target, single_gpu, args.max_retries
                            )
                            outcomes.append((value, 0, record, None))
                        except Exception as exc:
                            outcomes.append((None, 0, record, exc))
                for output, batch_index, record, target_error in outcomes:
                    if target_error is not None:
                        record_error(
                            failure(record["source_index"], "target", target_error)
                        )
                        continue
                    try:
                        payload = sample_payload(
                            output,
                            batch_index,
                            record["source_index"],
                            stored_layer_ids,
                            final_norm_layer_id,
                        )
                        index_metadata = sample_index_metadata(output, batch_index)
                        if is_tp_rank_0():
                            writer.add_sample(
                                record["source_index"],
                                payload,
                                sample_metadata=index_metadata,
                            )
                        shard_successes += 1
                        if shard_successes == args.shard_size:
                            commit_writer()
                    except Exception as exc:
                        record_error(
                            failure(record["source_index"], "serialization", exc)
                        )
            finally:
                del gpu_batch
                torch.cuda.empty_cache()

        for source_index in range(cursor, end):
            item = processed[source_index]
            preprocessing_error = item.get("preprocessing_error", "")
            if preprocessing_error:
                handle_batch(pending_records)
                pending_records = []
                record_error(
                    {
                        "source_index": source_index,
                        "stage": "preprocessing",
                        "error_type": preprocessing_error.split(":", 1)[0],
                        "error": preprocessing_error,
                    }
                )
            else:
                pending_records.append(
                    {
                        "source_index": source_index,
                        "input_ids": item["input_ids"],
                        "attention_mask": item["attention_mask"],
                        "loss_mask": item["loss_mask"],
                    }
                )
            if len(pending_records) >= args.batch_size:
                handle_batch(pending_records)
                pending_records = []
        handle_batch(pending_records)
        if is_tp_rank_0() and writer is not None and writer.consumed_start is not None:
            writer.commit({"rank": dp_rank, "final_local": True})
        elif is_tp_rank_0() and writer is not None:
            writer.abort()

        dist.barrier()
        if is_tp_rank_0():
            local_staging = discover_staging(output_root, identity, dp_rank)
            write_rank_errors(output_root, dp_rank, local_staging)
        dist.barrier()

        manifest = None
        if dist.get_rank() == 0:
            run_metadata = {
                "save_mode": "regen_full",
                "sample_index_version": SAMPLE_INDEX_VERSION,
                "target_model": args.target_model_path,
                "target_backend": args.target_model_backend,
                "chat_template": args.chat_template,
                "max_length": args.max_length,
                "requested_hidden_layer_ids": requested_layer_ids,
                "stored_hidden_layer_ids": stored_layer_ids,
                "final_norm_layer_id": final_norm_layer_id,
                "num_hidden_layers": num_hidden_layers,
                "hidden_size": hidden_size,
                "hidden_dtype": str(
                    getattr(config, "dtype", None)
                    or getattr(config, "torch_dtype", None)
                ),
                "compressed": args.compress,
            }
            manifest = finalize_staging(
                output_root, identity, args.shard_size, args.compress, run_metadata
            )
            print(
                f"wrote manifest: {output_root / 'manifest.json'} "
                f"successful={manifest.get('successful_samples')} "
                f"failed={manifest.get('failed_samples')}"
            )
        dist.barrier()
        return manifest
    finally:
        destroy_distributed()


def main(argv: list[str] | None = None) -> int:
    try:
        run_generation(parse_args(argv))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
