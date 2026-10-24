"""Generate versioned DLite token/hidden-state tar shards.

SGLang captures packed true-length sequences and writes asynchronous
``tar-raw-v1`` shards. Shard rotation uses ``--target-shard-mib``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA_VERSION = 2
SAMPLE_INDEX_VERSION = 1


class HiddenStateError(RuntimeError):
    pass


def parse_layer_ids(value: str | None) -> list[int] | None:
    if value is None:
        return None
    try:
        result = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "layer IDs must be comma-separated integers"
        ) from exc
    if (
        not result
        or len(set(result)) != len(result)
        or any(item < 0 for item in result)
    ):
        raise argparse.ArgumentTypeError(
            "layer IDs must be unique non-negative integers"
        )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    model = parser.add_argument_group("model")
    model.add_argument("--target-model-path", required=True)
    model.add_argument("--target-layer-ids", type=parse_layer_ids)
    model.add_argument("--trust-remote-code", action="store_true")
    model.add_argument("--model-download-dir")

    data = parser.add_argument_group("data")
    data.add_argument("--data-path", required=True)
    data.add_argument(
        "--output-path",
        help=(
            "Full-cache output directory. Defaults to "
            "./cache/data/regen_full/<input>_<model>."
        ),
    )
    data.add_argument("--chat-template", default="qwen")
    data.add_argument("--is-preformatted", action="store_true")
    data.add_argument("--max-length", type=int, default=4096)
    data.add_argument("--num-samples", type=int)
    data.add_argument("--build-dataset-num-proc", type=int, default=8)
    data.add_argument("--cache-dir", default="./cache/hidden_states")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--tp-size", type=int, default=1)
    runtime.add_argument("--batch-size", type=int, default=1)
    runtime.add_argument("--dist-timeout", type=int, default=2000)
    runtime.add_argument("--resume", action="store_true")
    runtime.add_argument(
        "--target-shard-mib",
        type=int,
        default=2048,
        help="close a tar shard once it reaches this size, in MiB",
    )
    runtime.add_argument("--queue-depth", type=int, default=2)
    runtime.add_argument("--writer-workers", type=int, default=2)
    runtime.add_argument(
        "--pending-mib",
        type=int,
        default=8192,
        help="pinned memory for copies waiting to be written, in MiB",
    )
    runtime.add_argument("--token-budget", type=int, default=0)
    runtime.add_argument("--length-bucketing", action="store_true")

    sglang = parser.add_argument_group("sglang backend")
    sglang.add_argument("--sglang-attention-backend", default="fa3")
    sglang.add_argument("--sglang-mem-fraction-static", type=float, default=0.3)
    sglang.add_argument("--sglang-context-length", type=int)
    sglang.add_argument("--sglang-enable-nccl-nvls", action="store_true")
    sglang.add_argument("--sglang-enable-symm-mem", action="store_true")
    sglang.add_argument("--sglang-enable-torch-compile", action="store_true")
    sglang.add_argument("--sglang-enable-dp-attention", action="store_true")
    sglang.add_argument("--sglang-enable-dp-lm-head", action="store_true")
    sglang.add_argument("--sglang-enable-piecewise-cuda-graph", action="store_true")
    sglang.add_argument(
        "--sglang-piecewise-cuda-graph-max-tokens", type=int, default=4096
    )
    sglang.add_argument("--sglang-piecewise-cuda-graph-tokens", type=int, nargs="+")
    sglang.add_argument("--sglang-ep-size", type=int, default=1)
    sglang.add_argument("--sglang-max-running-requests", type=int)
    sglang.add_argument("--sglang-max-total-tokens", type=int)
    args = parser.parse_args(argv)
    args.target_shard_bytes = args.target_shard_mib * 1024**2
    args.pending_bytes = args.pending_mib * 1024**2
    for name in (
        "max_length",
        "batch_size",
        "tp_size",
        "build_dataset_num_proc",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_samples is not None and args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if (
        min(
            args.target_shard_bytes,
            args.queue_depth,
            args.writer_workers,
            args.pending_bytes,
        )
        <= 0
        or args.token_budget < 0
    ):
        parser.error(
            "pipeline resource limits must be positive; token budget may be zero"
        )
    return args


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def tokenizer_fingerprint(tokenizer: Any) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    vocabulary = (
        backend.to_str()
        if backend is not None
        else json.dumps(tokenizer.get_vocab(), sort_keys=True)
    )
    return hashlib.sha256(
        (vocabulary + repr(tokenizer.chat_template)).encode()
    ).hexdigest()



def _path_component(value: str) -> str:
    """Make a model or dataset name safe to use as one path component."""
    safe_component = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )
    return safe_component.strip("._-") or "data"


def default_output_path(data_path: str, target_model_path: str) -> Path:
    dataset = _path_component(Path(data_path).stem)
    model = _path_component(target_model_path.rstrip("/\\").split("/")[-1])
    return Path("./cache/data/regen_full") / f"{dataset}_{model}"


def validate_packed_capture_directory(output_root: Path) -> None:
    """Do not append packed rows to an interrupted padded or unlabelled run."""
    marker = output_root / "capture_config.json"
    if any(output_root.iterdir()) and (
        not marker.exists()
        or json.loads(marker.read_text()).get("capture_request_layout") != "packed"
    ):
        raise HiddenStateError(
            "SGLang capture requires a packed capture_config.json; "
            "use a new output directory for a legacy or padded run"
        )


def run_generation(args: argparse.Namespace) -> dict[str, Any] | None:
    import torch.distributed as dist
    from datasets import load_dataset
    from transformers import AutoConfig, AutoTokenizer

    from litespec.args import SGLangBackendArgs
    from litespec.data import build_training_dataset
    from litespec.data.capture_pipeline import run_packed_capture
    from litespec.distributed import destroy_distributed, init_distributed
    from litespec.modeling.target import get_training_target_model

    input_path = Path(args.data_path)
    if not input_path.is_file():
        raise HiddenStateError(f"data path does not exist: {input_path}")
    output_root = (
        Path(args.output_path)
        if args.output_path
        else default_output_path(args.data_path, args.target_model_path)
    )
    identity = input_fingerprint(input_path)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = output_root / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != SCHEMA_VERSION:
                raise HiddenStateError(
                    f"existing cache schema is not {SCHEMA_VERSION}; "
                    "regenerate the cache"
                )
            if manifest.get("input_identity") != identity:
                raise HiddenStateError(
                    f"existing manifest belongs to another input: {manifest_path}"
                )
            if manifest.get("sample_index_version") != SAMPLE_INDEX_VERSION:
                raise HiddenStateError(
                    "existing cache has no lightweight sample index; regenerate it"
                )
            expected_fields = {
                "target_model": args.target_model_path,
                "target_backend": "sglang",
                "max_length": args.max_length,
                "chat_template": args.chat_template,
                "cache_format": "tar-raw-v1",
                "capture_request_layout": "packed",
            }
            for key, expected in expected_fields.items():
                actual = manifest.get(key)
                if actual != expected:
                    raise HiddenStateError(
                        f"resume configuration mismatch for {key}: {actual!r} != {expected!r}"
                    )
            if args.target_layer_ids:
                expected_layers = args.target_layer_ids
                if expected_layers != manifest.get("requested_hidden_layer_ids"):
                    raise HiddenStateError("resume target layer selection changed")
            if args.num_samples is not None and args.num_samples != manifest.get(
                "successful_samples", 0
            ) + manifest.get("failed_samples", 0):
                raise HiddenStateError("resume sample limit changed")
            if args.resume and manifest.get("complete"):
                if dist.get_rank() == 0:
                    print(
                        f"manifest already complete: {manifest_path} "
                        f"successful={manifest.get('successful_samples')} "
                        f"failed={manifest.get('failed_samples')}"
                    )
                return manifest if dist.get_rank() == 0 else None
        if not args.resume and any(output_root.iterdir()):
            raise HiddenStateError(
                f"output directory is not empty; use --resume: {output_root}"
            )
        validate_packed_capture_directory(output_root)
        dist.barrier()
        if dist.get_rank() == 0:
            atomic_write_text(
                output_root / "capture_config.json",
                json.dumps({"capture_request_layout": "packed"}) + "\n",
            )
        dist.barrier()
        for temp in output_root.glob("**/.*.tmp"):
            temp.unlink(missing_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(
            args.target_model_path,
            cache_dir=args.model_download_dir,
            trust_remote_code=args.trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = (
                tokenizer.eos_token_id or tokenizer.unk_token_id or 0
            )
        config = AutoConfig.from_pretrained(
            args.target_model_path,
            cache_dir=args.model_download_dir,
            trust_remote_code=args.trust_remote_code,
        )
        config = getattr(config, "text_config", config)
        target_kwargs = {
            "trust_remote_code": args.trust_remote_code,
            **SGLangBackendArgs.from_args(args).to_kwargs(),
        }
        target = get_training_target_model(
            args.target_model_path,
            backend="sglang",
            torch_dtype=getattr(config, "dtype", getattr(config, "torch_dtype", None)),
            device=None,
            cache_dir=args.model_download_dir,
            **target_kwargs,
        )
        num_hidden_layers = int(config.num_hidden_layers)
        hidden_size = int(config.hidden_size)
        final_norm_layer_id = num_hidden_layers - 1
        requested_layer_ids = (
            list(range(num_hidden_layers))
            if args.target_layer_ids is None
            else list(args.target_layer_ids)
        )
        invalid_layer_ids = [
            layer_id
            for layer_id in requested_layer_ids
            if not 0 <= layer_id < num_hidden_layers
        ]
        if invalid_layer_ids:
            raise HiddenStateError(
                f"target layer IDs {invalid_layer_ids} are outside [0, "
                f"{num_hidden_layers - 1}]"
            )
        stored_layer_ids = sorted(set(requested_layer_ids) | {final_norm_layer_id})
        target.set_capture_layers(stored_layer_ids)

        raw_dataset = load_dataset("json", data_files=str(input_path))["train"]
        if args.num_samples is not None:
            raw_dataset = raw_dataset.select(
                range(min(args.num_samples, len(raw_dataset)))
            )
        cache_key = hashlib.sha256(
            json.dumps(
                {
                    "identity": identity,
                    "num_samples": args.num_samples,
                    "tokenizer": tokenizer_fingerprint(tokenizer),
                    "preprocessing_version": 2,
                    "max_length": args.max_length,
                    "chat_template": args.chat_template,
                    "model": args.target_model_path,
                    "preformatted": args.is_preformatted,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

        def build_processed_dataset():
            return build_training_dataset(
                dataset=raw_dataset,
                tokenizer=tokenizer,
                chat_template=args.chat_template,
                max_length=args.max_length,
                shuffle_seed=None,
                num_proc=args.build_dataset_num_proc,
                cache_dir=str(Path(args.cache_dir) / "processed_dataset"),
                cache_key=cache_key,
                is_preformatted=args.is_preformatted,
                capture_errors=True,
            )

        processed = build_processed_dataset() if dist.get_rank() == 0 else None
        dist.barrier()
        if processed is None:
            processed = build_processed_dataset()

        args.output_path = str(output_root)
        return run_packed_capture(
            args,
            target,
            processed,
            identity,
            {
                "target_model": args.target_model_path,
                "target_backend": "sglang",
                "chat_template": args.chat_template,
                "max_length": args.max_length,
                "requested_hidden_layer_ids": requested_layer_ids,
                "stored_hidden_layer_ids": stored_layer_ids,
                "final_norm_layer_id": final_norm_layer_id,
                "num_hidden_layers": num_hidden_layers,
                "hidden_size": hidden_size,
                "hidden_dtype": str(target.model_runner.model_config.dtype),
                "compressed": False,
                "capture_request_layout": "packed",
            },
        )

    finally:
        destroy_distributed()


def main(argv: list[str] | None = None) -> int:
    try:
        run_generation(parse_args(argv))
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
