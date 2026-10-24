"""Bounded packed capture, D2H and persistence pipeline for TP owner ranks."""

from __future__ import annotations

import queue
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from .capture_storage import DirectTarWriter, publish_json, tensor_digest
from .hidden_raw import RawPayload, encode_raw_sample


@dataclass
class PinnedBlock:
    aux: torch.Tensor
    final: torch.Tensor
    valid: torch.Tensor

    @property
    def nbytes(self) -> int:
        return (self.aux.numel() + self.final.numel()) * self.aux.element_size() + 1


class PinnedPool:
    def __init__(self, byte_limit: int) -> None:
        self.byte_limit = byte_limit
        self.allocated = 0
        self.free: list[PinnedBlock] = []
        self.lock = threading.Condition()

    def acquire(
        self,
        tokens: int,
        aux_width: int,
        hidden_size: int,
        dtype: torch.dtype,
        check=None,
    ) -> PinnedBlock:
        capacity = (tokens + 255) // 256 * 256
        required = (
            capacity
            * (aux_width + hidden_size)
            * torch.empty((), dtype=dtype).element_size()
            + 1
        )
        with self.lock:
            for index, block in enumerate(self.free):
                if (
                    block.aux.shape[0] >= tokens
                    and block.aux.shape[1] == aux_width
                    and block.aux.dtype == dtype
                ):
                    return self.free.pop(index)
            if required > self.byte_limit:
                raise MemoryError("one capture batch exceeds the pinned byte limit")
            while self.allocated + required > self.byte_limit:
                while self.free and self.allocated + required > self.byte_limit:
                    self.allocated -= self.free.pop().nbytes
                if self.allocated + required <= self.byte_limit:
                    break
                if check is not None:
                    check()
                self.lock.wait(timeout=0.1)
            block = PinnedBlock(
                torch.empty((capacity, aux_width), dtype=dtype, pin_memory=True),
                torch.empty((capacity, hidden_size), dtype=dtype, pin_memory=True),
                torch.empty((), dtype=torch.bool, pin_memory=True),
            )
            self.allocated += block.nbytes
            return block

    def release(self, block: PinnedBlock) -> None:
        with self.lock:
            self.free.append(block)
            self.lock.notify_all()


@dataclass
class PendingBatch:
    output: Any
    records: list[dict[str, Any]]
    block: PinnedBlock
    event: torch.cuda.Event


def index_metadata(record: dict[str, Any], length: int) -> dict[str, Any]:
    active = record["loss_mask"].reshape(-1)[:length].gt(0.5).to(torch.int8)
    transitions = torch.cat([active, torch.zeros(1, dtype=torch.int8)]) - torch.cat(
        [torch.zeros(1, dtype=torch.int8), active]
    )
    starts = torch.nonzero(transitions == 1).reshape(-1).tolist()
    ends = torch.nonzero(transitions == -1).reshape(-1).tolist()
    return {
        "sequence_length": length,
        "supervised_token_count": int(active.sum()),
        "loss_mask_runs": [[start, end] for start, end in zip(starts, ends)],
        "token_digests": {
            key: tensor_digest(record[key].reshape(-1)[:length])
            for key in ("input_ids", "attention_mask", "loss_mask")
        },
    }


class CapturePipeline:
    def __init__(
        self, args: Any, root: Path, rank: int, identity: dict[str, Any]
    ) -> None:
        self.args = args
        self.pool = PinnedPool(args.pending_bytes)
        self.stream = torch.cuda.Stream()
        self.slots = threading.Semaphore(args.queue_depth)
        self.queue: queue.Queue[PendingBatch | None] = queue.Queue(args.queue_depth)
        self.error: BaseException | None = None
        self.writer = DirectTarWriter(
            root,
            rank,
            identity,
            args.target_shard_bytes,
            args.resume,
        )
        self.serializers = ThreadPoolExecutor(
            max_workers=args.writer_workers, thread_name_prefix="capture-serialize"
        )
        self.thread = threading.Thread(
            target=self._write_loop, name="capture-writer", daemon=True
        )
        self.thread.start()

    def check(self) -> None:
        if self.error is not None:
            raise RuntimeError("background capture writer failed") from self.error

    def reserve(self) -> None:
        while not self.slots.acquire(timeout=0.1):
            self.check()
        self.check()

    def submit(self, output: Any, records: list[dict[str, Any]]) -> None:
        if output.aux_hidden.dtype != output.final_hidden.dtype:
            raise ValueError(
                "capture layers have different dtypes; conversion is forbidden"
            )
        block = self.pool.acquire(
            output.offsets[-1],
            output.aux_hidden.shape[1],
            output.final_hidden.shape[1],
            output.final_hidden.dtype,
            self.check,
        )
        valid = (
            torch.isfinite(output.aux_hidden).all()
            & torch.isfinite(output.final_hidden).all()
        )
        ready = torch.cuda.Event()
        ready.record()
        done = torch.cuda.Event()
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(ready)
            block.aux[: output.offsets[-1]].copy_(output.aux_hidden, non_blocking=True)
            block.final[: output.offsets[-1]].copy_(
                output.final_hidden, non_blocking=True
            )
            block.valid.copy_(valid, non_blocking=True)
            done.record(self.stream)
            output.aux_hidden.record_stream(self.stream)
            output.final_hidden.record_stream(self.stream)
            valid.record_stream(self.stream)
        self.queue.put(PendingBatch(output, records, block, done))

    def _encode(
        self, pending: PendingBatch, index: int
    ) -> tuple[int, RawPayload, dict[str, Any]]:
        output = pending.output
        begin, end = output.offsets[index : index + 2]
        record = pending.records[index]
        tokens = {
            key: record[key].reshape(-1)[: end - begin].contiguous()
            for key in ["input_ids", "attention_mask", "loss_mask"]
        }
        aux = pending.block.aux[begin:end]
        final = pending.block.final[begin:end]
        payload = encode_raw_sample(
            record["source_index"],
            tokens,
            aux,
            final,
            output.layer_ids,
            output.final_layer_id,
        )
        metadata = index_metadata(record, end - begin)
        metadata.update(
            hidden_dtype=str(final.dtype),
            hidden_size=final.shape[1],
            stored_hidden_layer_ids=output.layer_ids + [output.final_layer_id],
        )
        return record["source_index"], payload, metadata

    def _write_loop(self) -> None:
        try:
            while True:
                pending = self.queue.get()
                if pending is None:
                    self.queue.task_done()
                    break
                try:
                    pending.event.synchronize()
                    if not bool(pending.block.valid):
                        raise ValueError(
                            "capture contains NaN or Inf; batch is not committed"
                        )
                    futures = [
                        self.serializers.submit(self._encode, pending, index)
                        for index in range(len(pending.records))
                    ]
                    try:
                        for future in futures:
                            source_index, payload, metadata = future.result()
                            self.writer.append(source_index, payload, metadata)
                    except BaseException as error:
                        self.error = error
                        raise
                    finally:
                        # Serializers retain views into this pinned block. Join every
                        # reader before making the block available to the producer.
                        from concurrent.futures import wait

                        wait(futures)
                finally:
                    self.pool.release(pending.block)
                    self.slots.release()
                    self.queue.task_done()
            self.writer.close()
        except BaseException as error:
            self.error = error
            self.writer.abort()
        finally:
            self.serializers.shutdown(wait=True)
            # Release producer waits after a failure. Unwritten batches stay uncommitted.
            self.slots.release()

    def close(self) -> list[dict[str, Any]]:
        while self.thread.is_alive():
            self.check()
            try:
                self.queue.put(None, timeout=0.1)
                break
            except queue.Full:
                continue
        self.thread.join()
        self.check()
        return self.writer.committed


def capture_records(
    processed: Any, indices: list[int], batch_size: int, token_budget: int, bucket: bool
):
    if bucket:
        indices = sorted(
            indices, key=lambda index: len(processed[index]["input_ids"].reshape(-1))
        )
    batch = []
    tokens = 0
    for index in indices:
        record = dict(processed[index])
        if record.get("preprocessing_error"):
            raise ValueError(
                f'preprocessing failed for sample {index}: {record["preprocessing_error"]}'
            )
        record["source_index"] = index
        length = int(record["attention_mask"].sum())
        if length <= 0 or length != record["input_ids"].numel():
            raise ValueError("packed requests require non-padded, nonempty records")
        if batch and (
            len(batch) >= batch_size or token_budget and tokens + length > token_budget
        ):
            yield batch
            batch, tokens = [], 0
        batch.append(record)
        tokens += length
    if batch:
        yield batch


def run_packed_capture(
    args: Any,
    target: Any,
    processed: Any,
    identity: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    from litespec.distributed import get_dp_group, get_tp_group, is_tp_rank_0

    root = Path(args.output_path)
    tp_group = get_tp_group()
    tp_ranks = dist.get_process_group_ranks(tp_group)
    # All world ranks create groups in the same order, including other DP replicas.
    cpu_group = None
    world_size = dist.get_world_size()
    world_cpu_group = dist.new_group(backend="gloo")
    for start in range(0, world_size, args.tp_size):
        ranks = list(range(start, start + args.tp_size))
        group = dist.new_group(ranks=ranks, backend="gloo")
        if dist.get_rank() in ranks:
            cpu_group = group
    owner = is_tp_rank_0()
    dp_rank, dp_size = dist.get_rank(get_dp_group()), dist.get_world_size(
        get_dp_group()
    )
    count, extra = divmod(len(processed), dp_size)
    begin = dp_rank * count + min(dp_rank, extra)
    end = begin + count + (dp_rank < extra)
    capture_identity = {
        "input": identity,
        "metadata": metadata,
        "batch_size": args.batch_size,
        "tp_size": args.tp_size,
        "attention_backend": args.sglang_attention_backend,
        "cache_format": "tar-raw-v1",
    }
    pipeline = None
    initial_error = None
    if owner:
        try:
            pipeline = CapturePipeline(args, root, dp_rank, capture_identity)
        except BaseException as exception:
            initial_error = "".join(traceback.format_exception(exception))
    state = [
        initial_error,
        pipeline.writer.completed_indices if pipeline is not None else None,
    ]
    dist.broadcast_object_list(state, src=tp_ranks[0], group=cpu_group)
    if state[0] is not None:
        raise RuntimeError(state[0])
    indices = [index for index in range(begin, end) if index not in state[1]]

    def coordinate(error: str | None) -> None:
        value = [error]
        dist.broadcast_object_list(value, src=tp_ranks[0], group=cpu_group)
        if value[0] is not None:
            raise RuntimeError(value[0])

    try:
        for records in capture_records(
            processed,
            indices,
            args.batch_size,
            args.token_budget,
            args.length_bucketing,
        ):
            error = None
            if owner:
                try:
                    pipeline.reserve()
                except BaseException as exception:
                    error = "".join(traceback.format_exception(exception))
            coordinate(error)
            token_lists = [
                record["input_ids"].reshape(-1).tolist() for record in records
            ]
            output = target.generate_capture_data(token_lists)
            if owner:
                try:
                    pipeline.submit(output, records)
                except BaseException as exception:
                    error = "".join(traceback.format_exception(exception))
            coordinate(error)
        local_shards = None
        error = None
        if owner:
            try:
                local_shards = pipeline.close()
            except BaseException as exception:
                error = "".join(traceback.format_exception(exception))
        coordinate(error)
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_shards, group=world_cpu_group)
        manifest = None
        final_error = None
        if dist.get_rank() == 0:
            try:
                shards = sorted(
                    [
                        shard
                        for rank_shards in gathered
                        if rank_shards is not None
                        for shard in rank_shards
                    ],
                    key=lambda shard: (shard["rank"], shard["sequence"]),
                )
                observed = [
                    index for shard in shards for index in shard["source_indices"]
                ]
                if len(observed) != len(set(observed)) or set(observed) != set(
                    range(len(processed))
                ):
                    raise ValueError("capture coverage is incomplete or duplicated")
                ordinal = 0
                for shard in shards:
                    shard["success_start"] = ordinal
                    ordinal += shard["sample_count"]
                    shard["success_end"] = ordinal - 1
                manifest = {
                    "schema_version": 2,
                    "sample_index_version": 1,
                    "complete": True,
                    "save_mode": "regen_full",
                    "cache_format": "tar-raw-v1",
                    "input_identity": identity,
                    "successful_samples": ordinal,
                    "failed_samples": 0,
                    "shards": shards,
                    **metadata,
                }
                publish_json(root / "manifest.json", manifest)
            except BaseException as exception:
                final_error = "".join(traceback.format_exception(exception))
        result = [final_error]
        dist.broadcast_object_list(result, src=0, group=world_cpu_group)
        if result[0] is not None:
            raise RuntimeError(result[0])
        return manifest
    finally:
        if pipeline is not None and pipeline.thread.is_alive():
            # On forward failure, stop background work before process-group teardown.
            try:
                pipeline.close()
            except BaseException:
                pass
