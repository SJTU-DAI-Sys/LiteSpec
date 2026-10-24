"""Abrupt-process interruption preserves committed legacy and direct tar data."""

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from litespec.data.capture_storage import DirectTarWriter
from litespec.data.hidden_raw import RawPayload

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/data/prepare_hidden_states.py"
spec = importlib.util.spec_from_file_location("capture_recovery_module", SCRIPT)
capture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = capture
spec.loader.exec_module(capture)

CHILD = r"""
import importlib.util, pathlib, sys, time
from litespec.data.capture_storage import DirectTarWriter
from litespec.data.hidden_raw import RawPayload
root, mode, script = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
if mode == 'direct':
    writer = DirectTarWriter(root, 0, {'fixture': 1}, 1024)
    writer.append(0, RawPayload(b'first', ()), {})
    writer.close()
    writer = DirectTarWriter(root, 0, {'fixture': 1}, 1024, resume=True)
    writer.append(1, RawPayload(b'uncommitted', ()), {})
else:
    spec = importlib.util.spec_from_file_location('capture_child', script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    writer = module.TarShardWriter(root / 'staging/rank_00000', 'stage', 0, False, 'staging', {'fixture': 1})
    writer.add_sample(0, b'first', sample_metadata={'sequence_length': 8, 'supervised_token_count': 8, 'loss_mask_runs': [[0, 8]]})
    writer.commit()
    writer = module.TarShardWriter(root / 'staging/rank_00000', 'stage', 1, False, 'staging', {'fixture': 1})
    writer.add_sample(1, b'uncommitted')
print('READY', flush=True)
while True:
    time.sleep(1)
"""


@pytest.mark.parametrize("mode", ["direct", "legacy"])
def test_abrupt_capture_resume(tmp_path, mode):
    child = subprocess.Popen(
        [sys.executable, "-u", "-c", CHILD, str(tmp_path), mode, str(SCRIPT)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "READY"
        child.kill()
        child.wait(timeout=10)
        assert child.returncode < 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    if mode == "direct":
        writer = DirectTarWriter(tmp_path, 0, {"fixture": 1}, 1024, resume=True)
        assert writer.completed_indices == {0}
        writer.append(1, RawPayload(b"second", ()), {})
        shards = writer.close()
        indices = [value for shard in shards for value in shard["source_indices"]]
        assert indices == [0, 1]
        for shard in shards:
            assert (
                hashlib.sha256((tmp_path / shard["path"]).read_bytes()).hexdigest()
                == shard["sha256"]
            )
    else:
        stages = capture.repair_staging(tmp_path, {"fixture": 1}, 0, 0)
        assert [
            value for shard in stages for value in shard.metadata["source_indices"]
        ] == [0]
        writer = capture.TarShardWriter(
            tmp_path / "staging/rank_00000",
            "stage",
            1,
            False,
            "staging",
            {"fixture": 1},
        )
        writer.add_sample(
            1,
            b"second",
            sample_metadata={
                "sequence_length": 8,
                "supervised_token_count": 8,
                "loss_mask_runs": [[0, 8]],
            },
        )
        writer.commit()
        manifest = capture.finalize_staging(
            tmp_path, {"fixture": 1}, 64, False, {"sample_index_version": 1}
        )
        assert manifest["complete"] and manifest["successful_samples"] == 2
        assert [
            value for shard in manifest["shards"] for value in shard["source_indices"]
        ] == [0, 1]
        # Simulate a crash after publishing the final partial tar and deleting
        # staging, but before publishing manifest. It must not recapture data.
        (tmp_path / "manifest.json").unlink()
        second = capture.finalize_staging(
            tmp_path, {"fixture": 1}, 64, False, {"sample_index_version": 1}
        )
        assert second["successful_samples"] == 2
