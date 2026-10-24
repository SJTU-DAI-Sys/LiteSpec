#!/usr/bin/env python3
"""Inspect and convert prompt datasets to FlashMTP regeneration JSONL.

The conversion engine is dataset-agnostic. Dataset-specific field selection lives
in a declarative JSON adapter or in a trusted Python adapter exposing
``extract(row)``.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

INPUT_FORMATS = ("auto", "json", "jsonl", "parquet", "hf")
TURN_MODES = ("multi", "first")
HF_FILTER_OPERATORS = {"=", "==", "!=", ">", ">=", "<", "<=", "in", "not in"}
DEFAULT_ROLE_MAP = {
    "system": "system",
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "tool": "tool",
}
PROMPT_FIELD_NAMES = {
    "prompt",
    "question",
    "instruction",
    "messages",
    "conversation",
    "conversations",
}


class ConversionError(ValueError):
    """Raised when input data or an adapter violates the conversion contract."""


def get_path(value: Any, path: str | None) -> Any:
    """Resolve a dotted mapping/list path such as ``payload.messages.0.role``."""
    if path is None or path == "":
        return value

    current = value
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                raise ConversionError(f"field path {path!r} is missing at {part!r}")
            current = current[part]
        elif isinstance(current, list) and part.isdigit():
            index = int(part)
            if index >= len(current):
                raise ConversionError(f"field path {path!r} has no list index {index}")
            current = current[index]
        else:
            raise ConversionError(f"field path {path!r} cannot traverse {part!r}")
    return current


def optional_path(value: Mapping[str, Any], path: str | None, default: Any) -> Any:
    if path is None:
        return default
    try:
        result = get_path(value, path)
    except ConversionError:
        return default
    return default if result is None else result


class JsonAdapter:
    """Declarative adapter for text prompts and ordinary message arrays."""

    def __init__(self, spec: Mapping[str, Any], adapter_path: Path) -> None:
        prompt = spec.get("prompt")
        if not isinstance(prompt, Mapping):
            raise ConversionError(f"{adapter_path}: 'prompt' must be an object")

        self.prompt_path = prompt.get("path")
        self.kind = prompt.get("kind")
        if not isinstance(self.prompt_path, str) or not self.prompt_path:
            raise ConversionError(
                f"{adapter_path}: prompt.path must be a non-empty string"
            )
        if self.kind not in {"text", "messages"}:
            raise ConversionError(
                f"{adapter_path}: prompt.kind must be 'text' or 'messages'"
            )

        self.role_path = prompt.get("role_path", "role")
        self.content_path = prompt.get("content_path", "content")
        if not isinstance(self.role_path, str) or not isinstance(
            self.content_path, str
        ):
            raise ConversionError(
                f"{adapter_path}: prompt.role_path and prompt.content_path must be strings"
            )

        configured_role_map = prompt.get("role_map", {})
        if not isinstance(configured_role_map, Mapping):
            raise ConversionError(f"{adapter_path}: prompt.role_map must be an object")
        self.role_map = dict(DEFAULT_ROLE_MAP)
        for raw_role, normalized_role in configured_role_map.items():
            if not isinstance(raw_role, str) or not isinstance(normalized_role, str):
                raise ConversionError(
                    f"{adapter_path}: role_map keys and values must be strings"
                )
            self.role_map[raw_role.strip().lower()] = normalized_role.strip().lower()

        self.source = self._metadata_spec(spec, "source", adapter_path)
        self.category = self._metadata_spec(spec, "category", adapter_path)

    @staticmethod
    def _metadata_spec(
        spec: Mapping[str, Any], name: str, adapter_path: Path
    ) -> tuple[str | None, Any]:
        value = spec.get(name, {})
        if value is None:
            return None, None
        if not isinstance(value, Mapping):
            raise ConversionError(f"{adapter_path}: {name!r} must be an object")
        path = value.get("path")
        if path is not None and not isinstance(path, str):
            raise ConversionError(
                f"{adapter_path}: {name}.path must be a string or null"
            )
        return path, value.get("default")

    def extract(self, row: Mapping[str, Any]) -> dict[str, Any]:
        raw_prompt = get_path(row, self.prompt_path)
        extracted: dict[str, Any]
        if self.kind == "text":
            extracted = {"prompt": raw_prompt}
        else:
            if not isinstance(raw_prompt, list):
                raise ConversionError(
                    f"prompt path {self.prompt_path!r} must contain a list of messages"
                )
            messages = []
            for index, message in enumerate(raw_prompt):
                if not isinstance(message, Mapping):
                    raise ConversionError(f"message {index} is not an object")
                raw_role = get_path(message, self.role_path)
                if not isinstance(raw_role, str) or not raw_role.strip():
                    raise ConversionError(f"message {index} has an invalid role")
                role_key = raw_role.strip().lower()
                if role_key not in self.role_map:
                    raise ConversionError(
                        f"message {index} has unmapped role {raw_role!r}"
                    )
                messages.append(
                    {
                        "role": self.role_map[role_key],
                        "content": get_path(message, self.content_path),
                    }
                )
            extracted = {"messages": messages}

        source_path, source_default = self.source
        category_path, category_default = self.category
        extracted["source"] = optional_path(row, source_path, source_default)
        extracted["category"] = optional_path(row, category_path, category_default)
        return extracted


def _load_python_module(path: Path) -> ModuleType:
    module_name = f"flashmtp_dataset_adapter_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ConversionError(f"cannot import Python adapter {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ConversionError(f"failed to import Python adapter {path}: {exc}") from exc
    return module


def load_adapter(path: str) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    adapter_path = Path(path)
    if not adapter_path.is_file():
        raise ConversionError(f"adapter does not exist: {adapter_path}")

    if adapter_path.suffix.lower() == ".json":
        try:
            with adapter_path.open(encoding="utf-8") as handle:
                spec = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ConversionError(
                f"failed to read JSON adapter {adapter_path}: {exc}"
            ) from exc
        if not isinstance(spec, Mapping):
            raise ConversionError(f"{adapter_path}: adapter root must be an object")
        return JsonAdapter(spec, adapter_path).extract

    if adapter_path.suffix.lower() == ".py":
        module = _load_python_module(adapter_path)
        extract = getattr(module, "extract", None)
        if not callable(extract):
            raise ConversionError(
                f"{adapter_path}: Python adapter must define extract(row)"
            )
        return extract

    raise ConversionError("adapter must end in .json or .py")


def infer_input_format(input_value: str, requested: str) -> str:
    if requested != "auto":
        return requested
    path = Path(input_value)
    if path.exists():
        if not path.is_file():
            raise ConversionError(
                "auto format only supports local files; use --input-format hf for a dataset ID"
            )
        suffix = path.suffix.lower()
        detected = {".json": "json", ".jsonl": "jsonl", ".parquet": "parquet"}.get(
            suffix
        )
        if detected:
            return detected
        raise ConversionError(f"cannot infer input format from extension {suffix!r}")
    return "hf"


def _require_local_file(input_value: str, input_format: str) -> Path:
    path = Path(input_value)
    if not path.is_file():
        raise ConversionError(f"{input_format} input is not a file: {path}")
    return path


def iter_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ConversionError(
                        f"{path}:{line_number}: invalid JSON: {exc}"
                    ) from exc
                if not isinstance(row, Mapping):
                    raise ConversionError(
                        f"{path}:{line_number}: each JSONL row must be an object"
                    )
                yield row
    except OSError as exc:
        raise ConversionError(f"failed to read {path}: {exc}") from exc


def iter_json(path: Path, records_path: str | None) -> Iterator[Mapping[str, Any]]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"failed to read JSON input {path}: {exc}") from exc
    records = get_path(value, records_path)
    if not isinstance(records, list):
        location = records_path or "<root>"
        raise ConversionError(f"JSON records at {location!r} must be an array")
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise ConversionError(f"JSON record {index} is not an object")
        yield row


def iter_parquet(path: Path) -> Iterator[Mapping[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ConversionError("Parquet input requires pyarrow") from exc
    try:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=1024):
            yield from batch.to_pylist()
    except Exception as exc:
        raise ConversionError(f"failed to read Parquet input {path}: {exc}") from exc


def _parse_hf_filter_value(raw_value: str) -> Any:
    """Parse JSON scalars/arrays while leaving ordinary CLI strings unchanged."""
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def normalize_hf_filters(
    raw_filters: list[list[str]] | None,
) -> list[tuple[str, str, Any]] | None:
    if not raw_filters:
        return None
    filters = []
    for column, operator, raw_value in raw_filters:
        column = column.strip()
        operator = operator.strip().lower()
        if not column:
            raise ConversionError("--hf-filter column must be non-empty")
        if operator not in HF_FILTER_OPERATORS:
            supported = ", ".join(sorted(HF_FILTER_OPERATORS))
            raise ConversionError(
                f"unsupported Hugging Face filter operator {operator!r}; "
                f"choose one of: {supported}"
            )
        value = _parse_hf_filter_value(raw_value)
        if operator in {"in", "not in"} and not isinstance(value, list):
            raise ConversionError(
                f"--hf-filter with operator {operator!r} requires a JSON array value"
            )
        filters.append((column, operator, value))
    return filters


def _load_hf_dataset(
    dataset_id: str,
    hf_config: str | None,
    split: str,
    hf_filters: list[tuple[str, str, Any]] | None = None,
) -> Iterable[Any]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ConversionError(
            "Hugging Face input requires the datasets package"
        ) from exc
    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if hf_config is not None:
        kwargs["name"] = hf_config
    if hf_filters is not None:
        kwargs["filters"] = hf_filters
    try:
        return load_dataset(dataset_id, **kwargs)
    except Exception as exc:
        raise ConversionError(
            f"failed to load Hugging Face dataset {dataset_id!r}: {exc}"
        ) from exc


def iter_records(
    input_value: str,
    input_format: str,
    records_path: str | None,
    hf_config: str | None,
    split: str,
    hf_filters: list[tuple[str, str, Any]] | None = None,
) -> Iterator[Mapping[str, Any]]:
    resolved_format = infer_input_format(input_value, input_format)
    if resolved_format != "json" and records_path is not None:
        raise ConversionError("--records-path only applies to JSON input")
    if resolved_format != "hf" and hf_config is not None:
        raise ConversionError("--hf-config only applies to Hugging Face input")
    if resolved_format != "hf" and hf_filters is not None:
        raise ConversionError("--hf-filter only applies to Hugging Face input")

    if resolved_format == "jsonl":
        yield from iter_jsonl(_require_local_file(input_value, resolved_format))
    elif resolved_format == "json":
        yield from iter_json(
            _require_local_file(input_value, resolved_format), records_path
        )
    elif resolved_format == "parquet":
        yield from iter_parquet(_require_local_file(input_value, resolved_format))
    elif resolved_format == "hf":
        for index, row in enumerate(
            _load_hf_dataset(input_value, hf_config, split, hf_filters)
        ):
            if not isinstance(row, Mapping):
                raise ConversionError(f"Hugging Face record {index} is not an object")
            yield row
    else:  # argparse constrains this, but keep the library surface defensive.
        raise ConversionError(f"unsupported input format: {resolved_format}")


def default_source(input_value: str, input_format: str) -> str:
    resolved_format = infer_input_format(input_value, input_format)
    return input_value if resolved_format == "hf" else Path(input_value).stem


def _normalize_content(content: Any, location: str) -> str:
    if not isinstance(content, str):
        raise ConversionError(
            f"{location} content must be text; multimodal content is unsupported"
        )
    content = content.strip()
    if not content:
        raise ConversionError(f"{location} content is empty")
    return content


def normalize_extracted(
    extracted: Any, fallback_source: str, turn_mode: str
) -> dict[str, Any]:
    if not isinstance(extracted, Mapping):
        raise ConversionError("adapter extract(row) must return an object")
    has_prompt = "prompt" in extracted
    has_messages = "messages" in extracted
    if has_prompt == has_messages:
        raise ConversionError(
            "adapter result must contain exactly one of 'prompt' or 'messages'"
        )

    if has_prompt:
        messages = [
            {
                "role": "user",
                "content": _normalize_content(extracted["prompt"], "prompt"),
            }
        ]
    else:
        raw_messages = extracted["messages"]
        if not isinstance(raw_messages, list):
            raise ConversionError("adapter result 'messages' must be a list")
        messages = []
        for index, message in enumerate(raw_messages):
            if not isinstance(message, Mapping):
                raise ConversionError(f"message {index} is not an object")
            role = message.get("role")
            if not isinstance(role, str):
                raise ConversionError(f"message {index} role must be text")
            role = role.strip().lower()
            if role not in {"system", "user", "assistant", "tool"}:
                raise ConversionError(
                    f"message {index} has unsupported normalized role {role!r}"
                )
            if role not in {"system", "user"}:
                continue
            raw_content = message.get("content")
            if isinstance(raw_content, str) and not raw_content.strip():
                continue
            content = _normalize_content(raw_content, f"message {index}")
            messages.append({"role": role, "content": content})

    if turn_mode == "first":
        first_user = next(
            (
                index
                for index, message in enumerate(messages)
                if message["role"] == "user"
            ),
            None,
        )
        if first_user is None:
            raise ConversionError("record has no non-empty user turn")
        leading_system = [
            message for message in messages[:first_user] if message["role"] == "system"
        ]
        messages = leading_system + [messages[first_user]]
    elif not any(message["role"] == "user" for message in messages):
        raise ConversionError("record has no non-empty user turn")

    source = extracted.get("source")
    if source is None or (isinstance(source, str) and not source.strip()):
        source = fallback_source
    elif not isinstance(source, str):
        source = str(source)
    else:
        source = source.strip()

    category = extracted.get("category")
    try:
        json.dumps(category, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ConversionError("category must be JSON-serializable") from exc

    return {"conversations": messages, "source": source, "category": category}


def _preview(value: Any, max_chars: int = 160) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        rendered = repr(value)
    return rendered if len(rendered) <= max_chars else rendered[: max_chars - 1] + "…"


def _field_descriptions(
    value: Any, path: str = "", depth: int = 0, max_depth: int = 3
) -> list[dict[str, str]]:
    descriptions: list[dict[str, str]] = []
    if path:
        descriptions.append(
            {"path": path, "type": type(value).__name__, "preview": _preview(value)}
        )
    if depth >= max_depth:
        return descriptions
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            descriptions.extend(
                _field_descriptions(child, child_path, depth + 1, max_depth)
            )
    elif isinstance(value, list) and value:
        child_path = f"{path}.0" if path else "0"
        descriptions.extend(
            _field_descriptions(value[0], child_path, depth + 1, max_depth)
        )
    return descriptions


def inspect_records(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    candidate_paths: dict[str, int] = {}
    for index, row in enumerate(
        iter_records(
            args.input,
            args.input_format,
            args.records_path,
            args.hf_config,
            args.split,
            normalize_hf_filters(args.hf_filter),
        )
    ):
        if index >= args.rows:
            break
        fields = _field_descriptions(row)
        rows.append({"row": index, "fields": fields})
        for field in fields:
            leaf = field["path"].split(".")[-1].lower()
            if leaf in PROMPT_FIELD_NAMES:
                candidate_paths[field["path"]] = (
                    candidate_paths.get(field["path"], 0) + 1
                )
    result = {
        "input": args.input,
        "input_format": infer_input_format(args.input, args.input_format),
        "rows_inspected": len(rows),
        "prompt_candidates": [
            path
            for path, _ in sorted(
                candidate_paths.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        "rows": rows,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def convert_records(args: argparse.Namespace) -> dict[str, Any]:
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise ConversionError(f"output already exists (use --overwrite): {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extract = load_adapter(args.adapter)
    fallback_source = default_source(args.input, args.input_format)
    processed = 0
    written = 0
    skipped = 0
    skip_reasons: dict[str, int] = {}
    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            records = iter_records(
                args.input,
                args.input_format,
                args.records_path,
                args.hf_config,
                args.split,
                normalize_hf_filters(args.hf_filter),
            )
            for row_index, row in enumerate(records):
                if args.limit is not None and processed >= args.limit:
                    break
                processed += 1
                try:
                    extracted = extract(row)
                    normalized = normalize_extracted(
                        extracted, fallback_source, args.turn_mode
                    )
                except Exception as exc:
                    if not args.skip_invalid:
                        if isinstance(exc, ConversionError):
                            raise ConversionError(f"record {row_index}: {exc}") from exc
                        raise ConversionError(
                            f"record {row_index}: adapter failed: {exc}"
                        ) from exc
                    reason = str(exc) or type(exc).__name__
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                    skipped += 1
                    print(f"skipping record {row_index}: {reason}", file=sys.stderr)
                    continue

                output = {
                    "id": written,
                    "conversations": normalized["conversations"],
                    "source": normalized["source"],
                    "category": normalized["category"],
                }
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                written += 1

        os.replace(temp_path, output_path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    summary = {
        "input": args.input,
        "output": str(output_path),
        "processed": processed,
        "written": written,
        "skipped": skipped,
        "skip_reasons": skip_reasons,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input", required=True, help="Local file path or Hugging Face dataset ID"
    )
    parser.add_argument("--input-format", choices=INPUT_FORMATS, default="auto")
    parser.add_argument(
        "--records-path", help="Dotted path to the record array in a JSON file"
    )
    parser.add_argument("--hf-config", help="Hugging Face dataset configuration name")
    parser.add_argument(
        "--hf-filter",
        action="append",
        nargs=3,
        metavar=("COLUMN", "OPERATOR", "VALUE"),
        help=(
            "Hugging Face Parquet filter; repeat for AND conditions. "
            "Values are parsed as JSON when possible."
        ),
    )
    parser.add_argument(
        "--split", default="train", help="Hugging Face split (default: train)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect or convert datasets to FlashMTP regeneration JSONL"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Print field paths and sample values"
    )
    add_source_arguments(inspect_parser)
    inspect_parser.add_argument(
        "--rows", type=int, default=5, help="Rows to inspect (default: 5)"
    )
    inspect_parser.set_defaults(handler=inspect_records)

    convert_parser = subparsers.add_parser(
        "convert", help="Convert records with an adapter"
    )
    add_source_arguments(convert_parser)
    convert_parser.add_argument(
        "--adapter", required=True, help="JSON or Python adapter path"
    )
    convert_parser.add_argument(
        "--output", required=True, help="Destination JSONL path"
    )
    convert_parser.add_argument("--turn-mode", choices=TURN_MODES, default="multi")
    convert_parser.add_argument(
        "--limit", type=int, help="Maximum number of input rows to process"
    )
    convert_parser.add_argument("--skip-invalid", action="store_true")
    convert_parser.add_argument("--overwrite", action="store_true")
    convert_parser.set_defaults(handler=convert_records)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "rows", 1) <= 0:
        parser.error("--rows must be positive")
    if getattr(args, "limit", None) is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    try:
        args.handler(args)
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        # Streaming Parquet readers can retain cyclic references to background
        # scanner resources until interpreter shutdown. Collect them while the
        # Python runtime is still fully initialized, especially after --limit.
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
