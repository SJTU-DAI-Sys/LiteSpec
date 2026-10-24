"""Keep capture defaults stable and reject unsupported pipeline combinations."""

import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

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


def test_public_capture_entrypoint_help():
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert "--capture-mode" in result.stdout
    assert "--profiling-output" not in result.stdout


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
        ("--capture-request-layout", "padded"),
        ("--capture-request-layout", "auto"),
        ("--capture-request-layout", "packed"),
        ("--profiling-output", "/tmp/capture_timings.json"),
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


def test_sync_capture_preserves_variable_length_samples_and_layers():
    from litespec.modeling.target.training_target_model import TargetCaptureOutput

    records = [
        {
            "source_index": index,
            "input_ids": torch.tensor(tokens),
            "attention_mask": torch.ones(len(tokens), dtype=torch.long),
            "loss_mask": torch.tensor(mask, dtype=torch.float32),
        }
        for index, tokens, mask in [
            (11, [1, 2, 3], [0, 1, 1]),
            (19, [4, 5, 6, 7, 8], [0, 0, 1, 1, 1]),
        ]
    ]
    aux = torch.arange(8 * 24, dtype=torch.float32).reshape(8, 24).bfloat16()
    final = torch.arange(8 * 12, dtype=torch.float32).reshape(8, 12).bfloat16()
    calls = []

    def capture_data(tokens):
        calls.append(tokens)
        return TargetCaptureOutput(aux, final, [0, 2], 3, [3, 5], [0, 3, 8])

    model = SimpleNamespace(generate_capture_data=capture_data)
    outputs = capture.capture_sglang_records(model, records, max_retries=0)
    assert calls == [[[1, 2, 3], [4, 5, 6, 7, 8]]]
    for index, (record, output) in enumerate(zip(records, outputs)):
        start, end = [0, 3, 8][index : index + 2]
        payload = torch.load(
            io.BytesIO(
                capture.sample_payload(output, 0, record["source_index"], [0, 2, 3], 3)
            ),
            weights_only=True,
        )
        assert payload["sample_index"] == record["source_index"]
        for key in ("input_ids", "attention_mask", "loss_mask"):
            assert torch.equal(payload[key], record[key])
        for column, layer in enumerate([0, 2]):
            assert torch.equal(
                payload["hidden_states"][layer],
                aux[start:end, column * 12 : (column + 1) * 12],
            )
        assert torch.equal(payload["final_hidden_state"], final[start:end])
        metadata = capture.sample_index_metadata(output, 0)
        assert metadata["sequence_length"] == len(record["input_ids"])
        assert metadata["loss_mask_runs"] == ([[1, 3]] if index == 0 else [[2, 5]])

    final[0, 0] = float("nan")
    with pytest.raises(capture.HiddenStateError, match="NaN or Inf"):
        capture.sample_payload(outputs[0], 0, 11, [0, 2, 3], 3)


def test_sync_capture_retries_backend_errors_and_rejects_wrong_lengths():
    calls = []

    def capture_data(tokens):
        calls.append(tokens)
        if len(calls) == 1:
            raise RuntimeError("transient capture failure")
        return SimpleNamespace(lengths=[4])

    model = SimpleNamespace(generate_capture_data=capture_data)
    with pytest.raises(capture.HiddenStateError, match="lengths do not match"):
        capture.capture_sglang_records(
            model, [{"input_ids": torch.tensor([1, 2, 3])}], max_retries=1
        )
    assert len(calls) == 2


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
