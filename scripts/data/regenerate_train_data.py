#!/usr/bin/env python3
"""Regenerate assistant turns in standard FlashMTP conversation JSONL."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from tqdm import tqdm


class RegenerationError(ValueError):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-file-path", required=True)
    parser.add_argument("--output-file-path")
    parser.add_argument("--server-address", required=True, nargs="+")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--is-reasoning-model", action="store_true")
    parser.add_argument("--is-gpt-oss", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.num_samples is not None and args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if not 0 <= args.temperature <= 2:
        parser.error("--temperature must be between 0 and 2")
    if args.max_tokens <= 0 or args.request_timeout <= 0:
        parser.error("--max-tokens and --request-timeout must be positive")
    return args


def normalize_server_address(address: str) -> str:
    address = address.rstrip("/")
    if not address.startswith(("http://", "https://")):
        address = f"http://{address}"
    return address


def dataset_name(path: str) -> str:
    name = re.sub(r"\.jsonl?$", "", Path(path).name)
    return re.sub(r"_\d+$", "", name)


def default_output_path(args: argparse.Namespace) -> Path:
    count = args.num_samples if args.num_samples is not None else "all"
    thinking = "on" if args.enable_thinking else "off"
    model = args.model.replace("/", "_").replace("\\", "_")
    filename = f"{dataset_name(args.input_file_path)}_think_{thinking}_samples_{count}_{model}_regen.jsonl"
    return Path("./cache/data/regen_token_only") / filename


def error_path_for(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_errors.jsonl")


def _record_key(value: Any) -> tuple[str, str]:
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise RegenerationError("record id must be a string or integer")
    return type(value).__name__, str(value)


def validate_record(value: Any, line_number: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RegenerationError(f"line {line_number}: record must be an object")
    required_field_names = {"id", "conversations", "source"}
    missing_field_names = required_field_names - value.keys()
    if missing_field_names:
        raise RegenerationError(
            f"line {line_number}: missing fields {sorted(missing_field_names)}"
        )
    _record_key(value["id"])
    if not isinstance(value["source"], str) or not value["source"].strip():
        raise RegenerationError(f"line {line_number}: source must be non-empty text")
    conversations = value["conversations"]
    if not isinstance(conversations, list) or not conversations:
        raise RegenerationError(f"line {line_number}: conversations must be non-empty")
    user_count = 0
    normalized = []
    for index, message in enumerate(conversations):
        if not isinstance(message, dict):
            raise RegenerationError(
                f"line {line_number}: message {index} is not an object"
            )
        role, content = message.get("role"), message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise RegenerationError(
                f"line {line_number}: message {index} has unsupported role {role!r}"
            )
        if not isinstance(content, str) or not content.strip():
            raise RegenerationError(
                f"line {line_number}: message {index} content must be non-empty text"
            )
        user_count += role == "user"
        normalized.append({"role": role, "content": content})
    if user_count == 0:
        raise RegenerationError(f"line {line_number}: conversation has no user turn")
    return {
        "id": value["id"],
        "conversations": normalized,
        "source": value["source"],
        "category": value.get("category"),
    }


def iter_records(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RegenerationError(
                    f"line {line_number}: invalid JSON: {exc}"
                ) from exc
            yield validate_record(value, line_number)


def load_processed_ids(paths: list[Path]) -> set[tuple[str, str]]:
    processed = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    key = _record_key(value["id"])
                except Exception as exc:
                    raise RegenerationError(
                        f"cannot resume: invalid record in {path}:{line_number}: {exc}"
                    ) from exc
                if key in processed:
                    raise RegenerationError(
                        f"duplicate processed id in resume files: {value['id']!r}"
                    )
                processed.add(key)
    return processed


def reasoning_effort() -> str:
    return random.choices(["low", "medium", "high"], weights=[4, 4, 2], k=1)[0]


def query_kwargs(
    args: argparse.Namespace, messages: list[dict], max_tokens: int | None = None
):
    kwargs = {
        "model": args.model,
        "messages": messages,
        "max_tokens": max_tokens if max_tokens is not None else args.max_tokens,
        "temperature": args.temperature,
        "stream": False,
        "timeout": args.request_timeout,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": args.enable_thinking}
        },
    }
    if args.top_p is not None:
        kwargs["top_p"] = args.top_p
    if args.repetition_penalty is not None:
        kwargs["presence_penalty"] = args.repetition_penalty
    if args.top_k is not None:
        kwargs["extra_body"]["top_k"] = args.top_k
    if args.is_gpt_oss:
        kwargs["reasoning_effort"] = reasoning_effort()
    return kwargs


def regenerate_record(
    args: argparse.Namespace, server_address: str, record: dict[str, Any]
) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(base_url=f"{server_address}/v1", api_key="None")
    regenerated = []
    try:
        for message in record["conversations"]:
            if message["role"] == "assistant":
                continue
            regenerated.append(message)
            if message["role"] != "user":
                continue
            response = client.chat.completions.create(**query_kwargs(args, regenerated))
            response_message = response.choices[0].message
            content = getattr(response_message, "content", None)
            if not isinstance(content, str) or not content:
                raise RegenerationError("server returned an empty assistant response")
            assistant = {"role": "assistant", "content": content}
            if args.is_reasoning_model:
                thinking = getattr(response_message, "reasoning_content", None)
                if thinking is not None:
                    assistant["thinking"] = thinking
            regenerated.append(assistant)
        return {**record, "conversations": regenerated}
    except Exception as exc:
        return {
            "id": record["id"],
            "source": record["source"],
            "category": record["category"],
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def check_server(args: argparse.Namespace, server_address: str) -> tuple[bool, str]:
    probe = {
        "id": "__healthcheck__",
        "conversations": [{"role": "user", "content": "Hi"}],
        "source": "healthcheck",
        "category": None,
    }
    old_timeout = args.request_timeout
    try:
        args.request_timeout = min(old_timeout, 30.0)
        result = regenerate_record(args, server_address, probe)
        return "error" not in result, result.get("error", "")
    finally:
        args.request_timeout = old_timeout


def _append_json(handle, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    handle.flush()


def regenerate(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input_file_path)
    if not input_path.is_file():
        raise RegenerationError(f"input does not exist: {input_path}")
    output_path = (
        Path(args.output_file_path)
        if args.output_file_path
        else default_output_path(args)
    )
    error_path = error_path_for(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.resume and (output_path.exists() or error_path.exists()):
        raise RegenerationError(
            f"output or error file exists; use --resume or choose another path: {output_path}"
        )
    processed = load_processed_ids([output_path, error_path]) if args.resume else set()

    servers = [normalize_server_address(address) for address in args.server_address]
    valid_servers = []
    for server in servers:
        healthy, error = check_server(args, server)
        if healthy:
            valid_servers.append(server)
        else:
            print(f"warning: server {server} is unavailable: {error}", file=sys.stderr)
    if not valid_servers:
        raise RegenerationError("no server passed the health check")

    mode = "a" if args.resume else "w"
    successes = failures = submitted = already_processed = 0
    seen_input: set[tuple[str, str]] = set()
    workers = args.concurrency * len(valid_servers)
    pending: dict[Future, Any] = {}

    def save_finished(done, output_handle, error_handle):
        nonlocal successes, failures
        for future in done:
            pending.pop(future, None)
            result = future.result()
            if "error" in result:
                _append_json(error_handle, result)
                failures += 1
            else:
                _append_json(output_handle, result)
                successes += 1

    with (
        output_path.open(mode, encoding="utf-8") as output_handle,
        error_path.open(mode, encoding="utf-8") as error_handle,
        ThreadPoolExecutor(max_workers=workers) as executor,
    ):
        progress = tqdm(desc="Regenerating", unit="sample")
        try:
            for record in iter_records(input_path):
                key = _record_key(record["id"])
                if key in seen_input:
                    raise RegenerationError(f"duplicate input id: {record['id']!r}")
                seen_input.add(key)
                if key in processed:
                    already_processed += 1
                    continue
                if args.num_samples is not None and submitted >= args.num_samples:
                    break
                while len(pending) >= workers:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    save_finished(done, output_handle, error_handle)
                    progress.update(len(done))
                server = valid_servers[submitted % len(valid_servers)]
                future = executor.submit(regenerate_record, args, server, record)
                pending[future] = record["id"]
                submitted += 1
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                save_finished(done, output_handle, error_handle)
                progress.update(len(done))
        finally:
            progress.close()

    summary = {
        "save_mode": "regen_token_only",
        "input": str(input_path),
        "output": str(output_path),
        "errors": str(error_path),
        "already_processed": already_processed,
        "submitted": submitted,
        "succeeded": successes,
        "failed": failures,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    try:
        regenerate(parse_args(argv))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
