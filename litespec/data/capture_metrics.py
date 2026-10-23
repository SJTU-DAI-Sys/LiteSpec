"""Low-overhead capture timings; CUDA events are collected after pipeline drain."""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Iterator


@dataclass
class StageTiming:
    seconds: float = 0.0
    calls: int = 0


class CaptureMetrics:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.stages: dict[str, StageTiming] = {}
        self.counters: dict[str, float] = {}
        self.events: list[tuple[str, Any, Any]] = []
        self.lock = threading.Lock()

    def add(self, name: str, seconds: float) -> None:
        with self.lock:
            timing = self.stages.setdefault(name, StageTiming())
            timing.seconds += seconds
            timing.calls += 1

    def count(self, name: str, value: float) -> None:
        with self.lock:
            self.counters[name] = self.counters.get(name, 0) + value

    @contextmanager
    def cpu(self, name: str) -> Iterator[None]:
        start = time.monotonic()
        try:
            yield
        finally:
            self.add(name, time.monotonic() - start)

    @contextmanager
    def cuda(self, name: str, stream: Any = None) -> Iterator[None]:
        import torch

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        try:
            yield
        finally:
            end.record(stream)
            with self.lock:
                self.events.append((name, start, end))

    def save(self, path: Path, extra: dict[str, Any] | None = None) -> None:
        import resource

        import torch

        # Called only after all writers have drained and persistence is complete.
        for name, start, end in self.events:
            end.synchronize()
            self.add(name, start.elapsed_time(end) / 1000)
        self.events.clear()
        value = {
            "elapsed_seconds": time.monotonic() - self.started,
            "stages": {key: asdict(item) for key, item in self.stages.items()},
            "counters": self.counters,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "proc_io": Path("/proc/self/io").read_text(),
            "overlapping_stages_are_not_additive": True,
            **(extra or {}),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w") as handle:
            json.dump(value, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)


metrics = CaptureMetrics()


def profile_cpu(name: str):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with metrics.cpu(name):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def persist_tree(root: Path) -> None:
    """Persist files and directory entries without reading file contents again."""
    with metrics.cpu("final_flush_fsync"):
        directories = [root]
        for path in root.rglob("*"):
            if path.is_dir():
                directories.append(path)
            elif path.is_file():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
        for path in reversed(directories):
            descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


def tensor_digest(value: Any) -> str:
    """Small token/mask digest recorded during serialization, without hidden rereads."""
    import hashlib

    import torch

    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()
