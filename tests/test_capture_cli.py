"""Keep capture defaults stable and reject unsupported pipeline combinations."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

script = Path(__file__).resolve().parents[1] / "scripts/data/prepare_hidden_states.py"
spec = importlib.util.spec_from_file_location("capture_cli_test_module", script)
capture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = capture
spec.loader.exec_module(capture)


def arguments(*extra):
    return capture.parse_args(
        [
            "--target-model-path",
            "/models/test",
            "--data-path",
            "/data/test.jsonl",
            *extra,
        ]
    )


def test_capture_defaults():
    args = arguments()
    assert args.batch_size == 1
    assert args.max_length == 4096 and args.token_budget == 0
    assert args.target_shard_mib == 2048 and args.pending_mib == 8192


def test_public_capture_entrypoint_help():
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert "--target-shard-mib" in result.stdout
    assert "--capture-mode" not in result.stdout
    assert "--cache-format" not in result.stdout
    assert "--shard-size" not in result.stdout
    assert "--profiling-output" not in result.stdout


def test_async_controls():
    args = arguments(
        "--tp-size",
        "2",
        "--batch-size",
        "4",
        "--token-budget",
        "8192",
    )
    assert args.tp_size == 2 and args.batch_size == 4 and args.token_budget == 8192
    assert args.queue_depth == 2 and args.writer_workers == 2
    assert args.target_shard_mib == 2048 and args.pending_mib == 8192
    assert args.target_shard_bytes == 2048 * 1024**2
    assert args.pending_bytes == 8192 * 1024**2


@pytest.mark.parametrize(
    "options",
    [
        ("--capture-mode", "async"),
        ("--cache-format", "tar-v2"),
        ("--shard-size", "512"),
        ("--queue-depth", "0"),
        ("--token-budget", "-1"),
        ("--capture-request-layout", "padded"),
        ("--profiling-output", "/tmp/capture_timings.json"),
    ],
)
def test_invalid_controls(options):
    with pytest.raises(SystemExit):
        arguments(*options)


def test_packed_resume_rejects_legacy_or_padded_partial_runs(tmp_path):
    capture.validate_packed_capture_directory(tmp_path)
    (tmp_path / "staging").mkdir()
    with pytest.raises(capture.HiddenStateError, match="new output directory"):
        capture.validate_packed_capture_directory(tmp_path)
    marker = tmp_path / "capture_config.json"
    marker.write_text(json.dumps({"capture_request_layout": "padded"}))
    with pytest.raises(capture.HiddenStateError, match="new output directory"):
        capture.validate_packed_capture_directory(tmp_path)
    marker.write_text(json.dumps({"capture_request_layout": "packed"}))
    capture.validate_packed_capture_directory(tmp_path)
