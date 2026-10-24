"""CUDA event lifetime and bounded pinned/queue resources (run in isolation)."""

import threading
import time
from types import SimpleNamespace

import pytest
import torch

from litespec.data.capture_pipeline import CapturePipeline, PinnedPool
from litespec.data.hidden_raw import decode_raw_sample
from litespec.modeling.target.training_target_model import TargetCaptureOutput

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def options():
    return SimpleNamespace(
        pending_bytes=128 * 1024**2,
        queue_depth=1,
        target_shard_bytes=8 * 1024**2,
        resume=False,
        writer_workers=2,
    )


def output(nan=False):
    aux = (
        torch.arange(16 * 24, device="cuda", dtype=torch.float32)
        .reshape(16, 24)
        .to(torch.bfloat16)
    )
    final = torch.ones(16, 12, device="cuda", dtype=torch.bfloat16)
    if nan:
        aux[0, 0] = float("nan")
    records = [
        {
            "source_index": 0,
            "input_ids": torch.arange(16),
            "attention_mask": torch.ones(16, dtype=torch.long),
            "loss_mask": torch.ones(16, dtype=torch.long),
        }
    ]
    return TargetCaptureOutput(aux, final, [0, 2], 3, [16], [0, 16]), records


def test_event_lifetime_backpressure_and_reuse(tmp_path, monkeypatch):
    pipeline = CapturePipeline(options(), tmp_path, 0, {})
    entered, release = threading.Event(), threading.Event()
    append = pipeline.writer.append

    def delayed(*args):
        entered.set()
        assert release.wait(10)
        return append(*args)

    monkeypatch.setattr(pipeline.writer, "append", delayed)
    pipeline.reserve()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        value, records = output()
        expected = value.aux_hidden.cpu()
        torch.cuda._sleep(100_000_000)
        pipeline.submit(value, records)
    del value
    waiter = threading.Thread(target=pipeline.reserve)
    waiter.start()
    assert entered.wait(10)
    assert waiter.is_alive() and not pipeline.pool.free
    release.set()
    waiter.join(10)
    assert not waiter.is_alive()
    pipeline.slots.release()
    shards = pipeline.close()
    assert pipeline.pool.allocated <= options().pending_bytes
    block = pipeline.pool.free[0]
    pointer = block.aux.data_ptr()
    assert pipeline.pool.acquire(16, 24, 12, torch.bfloat16).aux.data_ptr() == pointer
    member = shards[0]["members"][0]
    with (tmp_path / shards[0]["path"]).open("rb") as handle:
        handle.seek(member["offset"])
        restored = decode_raw_sample(handle.read(member["size"]))
    assert torch.equal(restored["hidden_states"][0], expected[:, :12])
    assert torch.equal(restored["hidden_states"][2], expected[:, 12:])


def test_nan_never_published(tmp_path):
    pipeline = CapturePipeline(options(), tmp_path, 0, {})
    pipeline.reserve()
    value, records = output(nan=True)
    pipeline.submit(value, records)
    with pytest.raises(RuntimeError, match="background capture writer failed"):
        pipeline.close()
    assert not list(tmp_path.glob("shards/rank_00000/*.json"))
    assert not (tmp_path / "manifest.json").exists()


def test_pinned_byte_credit_waits_for_release():
    required = 256 * (24 + 12) * 2 + 1
    pool = PinnedPool(required)
    first = pool.acquire(16, 24, 12, torch.bfloat16)
    result = []
    thread = threading.Thread(
        target=lambda: result.append(pool.acquire(16, 24, 12, torch.bfloat16))
    )
    thread.start()
    time.sleep(0.1)
    assert thread.is_alive() and pool.allocated <= required
    pool.release(first)
    thread.join(5)
    assert not thread.is_alive() and len(result) == 1 and pool.allocated <= required


def test_failed_serializer_joins_other_readers_before_buffer_release(
    tmp_path, monkeypatch
):
    pipeline = CapturePipeline(options(), tmp_path, 0, {})
    entered, release = threading.Event(), threading.Event()
    value, records = output()
    value.aux_hidden = torch.cat([value.aux_hidden, value.aux_hidden])
    value.final_hidden = torch.cat([value.final_hidden, value.final_hidden])
    value.lengths, value.offsets = [16, 16], [0, 16, 32]
    records = [records[0], {**records[0], "source_index": 1}]
    encode = pipeline._encode

    def injected(pending, index):
        if index == 0:
            assert entered.wait(10)
            raise OSError("serializer fixture failure")
        entered.set()
        assert release.wait(10)
        return encode(pending, index)

    monkeypatch.setattr(pipeline, "_encode", injected)
    pipeline.reserve()
    pipeline.submit(value, records)
    assert entered.wait(10)
    deadline = time.monotonic() + 5
    while pipeline.error is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert pipeline.error is not None and not pipeline.pool.free
    release.set()
    pipeline.thread.join(10)
    with pytest.raises(RuntimeError, match="background capture writer failed"):
        pipeline.close()
    assert not list(tmp_path.glob("shards/rank_00000/*.json"))
