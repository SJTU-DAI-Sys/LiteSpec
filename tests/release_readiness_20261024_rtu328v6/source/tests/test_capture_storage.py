"""Lossless raw layout, atomic publication, bounded commit and resume fixtures."""

import hashlib
import json
import tarfile
import threading
import time

import pytest
import torch

from litespec.data.capture_storage import DirectTarWriter, StreamingTar
from litespec.data.hidden_cache import RegenFullDataset, load_hidden_cache_manifest
from litespec.data.hidden_raw import decode_raw_sample, encode_raw_sample


def sample(index=0, width=12, layers=(0, 2)):
    tokens = {
        "input_ids": torch.arange(17),
        "attention_mask": torch.ones(17, dtype=torch.long),
        "loss_mask": torch.tensor([0] * 3 + [1] * 14),
    }
    aux = (
        torch.arange(17 * width * len(layers), dtype=torch.float32)
        .reshape(17, -1)
        .to(torch.bfloat16)
    )
    final = torch.randn(17, width, dtype=torch.bfloat16)
    return (
        encode_raw_sample(index, tokens, aux, final, list(layers), 3),
        tokens,
        aux,
        final,
    )


@pytest.mark.parametrize("width,layers", [(12, (0, 2)), (17, (1,)), (4, ())])
def test_raw_exact_views(width, layers):
    payload, tokens, aux, final = sample(width=width, layers=layers)
    restored = decode_raw_sample(b"".join(payload.chunks()))
    assert payload.size == len(b"".join(payload.chunks()))
    for key, value in tokens.items():
        assert torch.equal(value, restored[key])
        assert restored[key].dtype == value.dtype
    for position, layer in enumerate(layers):
        assert torch.equal(
            restored["hidden_states"][layer],
            aux[:, position * width : (position + 1) * width],
        )
    assert torch.equal(restored["final_hidden_state"], final)
    if len(layers) > 1:
        assert (
            restored["hidden_states"][layers[0]].untyped_storage().data_ptr()
            == restored["hidden_states"][layers[1]].untyped_storage().data_ptr()
        )


def test_descriptor_corruption():
    payload, *_ = sample()
    data = bytearray(b"".join(payload.chunks()))
    data[0] ^= 1
    with pytest.raises(ValueError):
        decode_raw_sample(data)
    with pytest.raises(ValueError):
        decode_raw_sample(b"".join(payload.chunks())[:-1])


def write_one(root):
    writer = DirectTarWriter(root, 0, {"fixture": 1}, 1024**2)
    payload, *_ = sample()
    metadata = {
        "sequence_length": 17,
        "supervised_token_count": 14,
        "loss_mask_runs": [[3, 17]],
    }
    writer.append(0, payload, metadata)
    return writer.close()


def test_tar_sha_reader_and_resume(tmp_path):
    shards = write_one(tmp_path)
    shard = shards[0]
    path = tmp_path / shard["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == shard["sha256"]
    with tarfile.open(path) as archive:
        member = archive.getmembers()[0]
        assert member.offset_data == shard["members"][0]["offset"]
        assert member.size == shard["members"][0]["size"]
        assert (
            decode_raw_sample(archive.extractfile(member).read())["sample_index"] == 0
        )
    shard.update(success_start=0, success_end=0)
    manifest = {
        "schema_version": 2,
        "sample_index_version": 1,
        "complete": True,
        "save_mode": "regen_full",
        "cache_format": "tar-raw-v1",
        "target_model": "fixture",
        "num_hidden_layers": 4,
        "hidden_size": 12,
        "stored_hidden_layer_ids": [0, 2, 3],
        "final_norm_layer_id": 3,
        "max_length": 32,
        "successful_samples": 1,
        "shards": shards,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    checked = load_hidden_cache_manifest(
        tmp_path,
        target_model="fixture",
        num_hidden_layers=4,
        hidden_size=12,
        required_layer_ids={0, 2},
        max_length=32,
        verify_checksums=True,
    )
    dataset = RegenFullDataset(
        tmp_path, checked, required_layer_ids={0, 2}, max_length=32
    )
    assert dataset[0]["hidden_states"][3].shape == (17, 12)
    assert dataset.eligible_indices(block_size=4) == [0]
    resumed = DirectTarWriter(tmp_path, 0, {"fixture": 1}, 1024**2, resume=True)
    assert resumed.completed_indices == {0}
    assert len(resumed.close()) == 1
    with path.open("r+b") as handle:
        handle.seek(1024)
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="corrupted"):
        DirectTarWriter(tmp_path, 0, {"fixture": 1}, 1024**2, resume=True)


def test_failed_publication_is_not_committed(tmp_path, monkeypatch):
    import litespec.data.capture_storage as storage

    original = storage.publish_json

    def fail(path, value):
        raise OSError("fixture sidecar failure")

    monkeypatch.setattr(storage, "publish_json", fail)
    writer = DirectTarWriter(tmp_path, 0, {"fixture": 1}, 1024**2)
    payload, *_ = sample()
    writer.append(0, payload, {})
    with pytest.raises(OSError, match="sidecar failure"):
        writer.close()
    assert not list(tmp_path.glob("shards/rank_00000/*.json"))
    monkeypatch.setattr(storage, "publish_json", original)
    resumed = DirectTarWriter(tmp_path, 0, {"fixture": 1}, 1024**2, resume=True)
    assert resumed.completed_indices == set()
    resumed.append(0, payload, {})
    assert resumed.close()[0]["source_indices"] == [0]


def test_commit_queue_is_bounded(tmp_path, monkeypatch):
    original = StreamingTar.commit
    entered = threading.Event()
    release = threading.Event()

    def slow(self, metadata):
        entered.set()
        assert release.wait(10)
        return original(self, metadata)

    monkeypatch.setattr(StreamingTar, "commit", slow)
    writer = DirectTarWriter(tmp_path, 0, {}, 1)
    payload, *_ = sample()
    errors = []

    def produce():
        try:
            for index in range(5):
                writer.append(index, payload, {})
            writer.close()
        except Exception as error:
            errors.append(error)

    producer = threading.Thread(target=produce)
    producer.start()
    assert entered.wait(5)
    time.sleep(0.1)
    assert producer.is_alive()
    assert len(writer.pending) <= 2
    release.set()
    producer.join(10)
    assert not producer.is_alive() and not errors
    assert len(writer.committed) == 5
