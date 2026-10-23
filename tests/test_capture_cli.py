"""Keep capture defaults stable and reject unsupported pipeline combinations."""

import importlib.util
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


def test_legacy_defaults():
    args = arguments()
    assert args.capture_mode == "sync" and args.cache_format == "tar-v2"
    assert args.batch_size == 1 and args.shard_size == 512
    assert args.max_length == 4096 and args.token_budget == 0


def test_async_controls():
    args = arguments(
        "--target-model-backend",
        "sglang",
        "--capture-mode",
        "async",
        "--cache-format",
        "tar-raw-v1",
        "--tp-size",
        "2",
        "--batch-size",
        "4",
        "--token-budget",
        "8192",
    )
    assert args.tp_size == 2 and args.batch_size == 4 and args.token_budget == 8192
    assert args.queue_depth == 2 and args.writer_workers == 2
    assert args.pending_bytes == 8 * 1024**3


@pytest.mark.parametrize(
    "options",
    [
        ("--capture-mode", "async"),
        ("--queue-depth", "0"),
        ("--token-budget", "-1"),
        (
            "--target-model-backend",
            "sglang",
            "--cache-format",
            "tar-raw-v1",
            "--compress",
        ),
    ],
)
def test_invalid_controls(options):
    with pytest.raises(SystemExit):
        arguments(*options)
