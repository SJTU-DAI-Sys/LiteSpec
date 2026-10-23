"""Streaming tar publication with bounded asynchronous durable commits."""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .capture_metrics import metrics
from .hidden_raw import RawPayload

BUFFER_SIZE = 4 * 1024 * 1024


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with metrics.cpu("index_publish"):
        with temporary.open("w") as handle:
            json.dump(value, handle, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)


class StreamingTar:
    def __init__(self, path: Path) -> None:
        self.final = path
        self.temporary = path.with_name("." + path.name + ".tmp")
        self.handle = self.temporary.open("wb", buffering=BUFFER_SIZE)
        self.digest = hashlib.sha256()
        self.size = 0

    def write(self, data: bytes | memoryview) -> None:
        for start in range(0, len(data), BUFFER_SIZE):
            chunk = data[start : start + BUFFER_SIZE]
            with metrics.cpu("tar_write"):
                self.handle.write(chunk)
            with metrics.cpu("sha_stream"):
                self.digest.update(chunk)
            self.size += len(chunk)
        metrics.count("application_written_bytes", len(data))

    def member(self, name: str, payload: RawPayload) -> dict[str, Any]:
        header = tarfile.TarInfo(name)
        header.size = payload.size
        self.write(header.tobuf(format=tarfile.USTAR_FORMAT))
        offset = self.size
        for chunk in payload.chunks():
            self.write(chunk)
        self.write(b"\0" * (-payload.size % 512))
        return {"name": name, "offset": offset, "size": payload.size}

    def seal(self) -> None:
        self.write(b"\0" * 1024)
        self.write(b"\0" * (-self.size % tarfile.RECORDSIZE))

    def commit(self, metadata: dict[str, Any]) -> dict[str, Any]:
        try:
            with metrics.cpu("flush_fsync"):
                self.handle.flush()
                os.fsync(self.handle.fileno())
                self.handle.close()
            with metrics.cpu("atomic_publish"):
                os.replace(self.temporary, self.final)
                fsync_directory(self.final.parent)
            result = {
                **metadata,
                "sha256": self.digest.hexdigest(),
                "size_bytes": self.size,
            }
            # The sidecar is the commit record. Orphaned data is never resumed.
            publish_json(self.final.with_suffix(".json"), result)
            return result
        finally:
            self.handle.close()

    def abort(self) -> None:
        self.handle.close()
        self.temporary.unlink(missing_ok=True)


class DirectTarWriter:
    """One ordered tar writer; only fsync/publication runs on the commit worker."""

    def __init__(
        self,
        root: Path,
        rank: int,
        identity: dict[str, Any],
        target_bytes: int,
        resume: bool = False,
        raw: bool = True,
    ) -> None:
        self.root = root
        self.directory = root / "shards" / f"rank_{rank:05d}"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.extension = "bin" if raw else "pt"
        self.identity = identity
        self.target_bytes = target_bytes
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tar-commit"
        )
        self.pending: list[Future] = []
        self.committed: list[dict[str, Any]] = []
        self.writer: StreamingTar | None = None
        self.entries: list[dict[str, Any]] = []
        self.sequence = 0
        if resume:
            for sidecar in sorted(self.directory.glob("part_*.json")):
                item = json.loads(sidecar.read_text())
                path = root / item["path"]
                if item["input_identity"] != identity:
                    raise ValueError("resume input identity mismatch")
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(BUFFER_SIZE), b""):
                        digest.update(chunk)
                if digest.hexdigest() != item["sha256"]:
                    raise ValueError(f"committed shard corrupted: {path}")
                self.committed.append(item)
                self.sequence = max(self.sequence, int(item["sequence"]) + 1)
        elif any(self.directory.iterdir()):
            raise ValueError("output contains data; use --resume")
        for temporary in self.directory.glob(".*.tmp"):
            temporary.unlink()

    @property
    def completed_indices(self) -> set[int]:
        return {
            int(index) for shard in self.committed for index in shard["source_indices"]
        }

    def check(self) -> None:
        while self.pending and self.pending[0].done():
            self.committed.append(self.pending.pop(0).result())

    def append(self, index: int, payload: RawPayload, metadata: dict[str, Any]) -> None:
        self.check()
        if (
            self.writer is not None
            and self.writer.size + payload.size > self.target_bytes
            and self.entries
        ):
            self.rotate()
        if self.writer is None:
            path = self.directory / f"part_{self.sequence:08d}.tar"
            self.writer = StreamingTar(path)
        member = self.writer.member(f"samples/{index:012d}.{self.extension}", payload)
        self.entries.append(
            {"source_index": index, "member": member, "metadata": metadata}
        )

    def rotate(self) -> None:
        if self.writer is None:
            return
        with metrics.cpu("commit_backpressure"):
            while len(self.pending) >= 2:
                self.committed.append(self.pending.pop(0).result())
        self.writer.seal()
        metadata = {
            "path": str(self.writer.final.relative_to(self.root)),
            "sequence": self.sequence,
            "rank": self.rank,
            "input_identity": self.identity,
            "source_indices": [entry["source_index"] for entry in self.entries],
            "sample_metadata": [entry["metadata"] for entry in self.entries],
            "members": [entry["member"] for entry in self.entries],
            "sample_count": len(self.entries),
        }
        self.pending.append(self.executor.submit(self.writer.commit, metadata))
        self.writer = None
        self.entries = []
        self.sequence += 1

    def close(self) -> list[dict[str, Any]]:
        try:
            self.rotate()
            with metrics.cpu("commit_final_drain"):
                for pending in self.pending:
                    self.committed.append(pending.result())
            self.pending.clear()
            return self.committed
        finally:
            self.executor.shutdown(wait=True)

    def abort(self) -> None:
        if self.writer is not None:
            self.writer.abort()
        self.executor.shutdown(wait=True)
