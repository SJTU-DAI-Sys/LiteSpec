"""Abrupt-process interruption preserves committed direct tar data."""

import hashlib
import subprocess
import sys

from litespec.data.capture_storage import DirectTarWriter
from litespec.data.hidden_raw import RawPayload

CHILD = r"""
import pathlib, sys, time
from litespec.data.capture_storage import DirectTarWriter
from litespec.data.hidden_raw import RawPayload
root = pathlib.Path(sys.argv[1])
writer = DirectTarWriter(root, 0, {'fixture': 1}, 1024)
writer.append(0, RawPayload(b'first', ()), {})
writer.close()
writer = DirectTarWriter(root, 0, {'fixture': 1}, 1024, resume=True)
writer.append(1, RawPayload(b'uncommitted', ()), {})
print('READY', flush=True)
while True:
    time.sleep(1)
"""


def test_abrupt_capture_resume(tmp_path):
    child = subprocess.Popen(
        [sys.executable, "-u", "-c", CHILD, str(tmp_path)],
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
