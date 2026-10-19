import argparse
import json
import random
import sys
import time
from collections import defaultdict
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from loguru import logger
import numpy as np
import torch
from rich import print
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from litespec.modeling.draft.dlite import DLiteDraftModel
from litespec.modeling.draft.dlite import sample
from litespec.modeling.kv_cache import make_target_cache

from evaluation import distributed as dist
from evaluation.model_loading import load_dlite_benchmark_models
from evaluation.utils import load_and_process_dataset

DATASET_PATH_FILE = Path(__file__).resolve().with_name("dataset_path.json")


def format_longbench_v2_prompt(data: dict) -> str:
    if "context" not in data:
        raise ValueError("Missing 'context' field in LongBench_v2 item")
    if "question" not in data:
        raise ValueError("Missing 'question' field in LongBench_v2 item")
    return f"{data['context']}\n\nQuestion: {data['question']}"


def is_longbench_v2_dataset_path(dataset_path: Path) -> bool:
    """Match LongBench v2 shards: ``longbench_v2`` or ``longbench_v2_*`` (e.g. ``64000_32000`` = 32k–64k)."""
    try:
        resolved = dataset_path.resolve()
    except OSError:
        resolved = dataset_path
    for part in resolved.parts:
        pl = part.lower()
        if pl == "longbench_v2" or pl.startswith("longbench_v2_"):
            return True
    return False


def load_longbench_v2_json_records(
    data: list, dataset_path: Path, dataset_alias: str = ""
) -> list[dict]:
    if not isinstance(data, list):
        raise ValueError(f"{dataset_path} must contain a JSON list")
    return [{"turns": [format_longbench_v2_prompt(item)]} for item in data]


def load_longbench_v2_shard_directory(
    shard_dir: Path, dataset_alias: str = ""
) -> list[dict]:
    json_files = sorted(p for p in shard_dir.glob("*.json") if p.is_file())
    if not json_files:
        raise FileNotFoundError(
            f"No JSON files under LongBench v2 shard dir: {shard_dir}"
        )
    instances: list[dict] = []
    for json_file in json_files:
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        instances.extend(load_longbench_v2_json_records(data, json_file, dataset_alias))
    if not instances:
        raise ValueError(f"No LongBench v2 samples loaded from shard dir: {shard_dir}")
    return instances


def is_multifieldqa_en_mixup_dataset_path(dataset_path: Path) -> bool:
    """LVEval mixup JSON: multifieldqa_en_mixup, lic_mixup, hotpotwikiqa_mixup (``context`` + ``input``)."""
    try:
        resolved = dataset_path.resolve()
    except OSError:
        resolved = dataset_path
    stem = resolved.stem.lower()
    markers = ("multifieldqa_en_mixup", "lic_mixup", "hotpotwikiqa_mixup")
    if any(m in stem for m in markers):
        return True
    for part in resolved.parts:
        pl = part.lower()
        if any(m in pl for m in markers):
            return True
    return False


def format_multifieldqa_en_mixup_prompt(data: dict) -> str:
    """LVEval: long ``context`` + ``input`` (question / instruction)."""
    if "context" not in data:
        raise ValueError("Missing 'context' field in multifieldqa_en_mixup item")
    if "input" not in data:
        raise ValueError("Missing 'input' field in multifieldqa_en_mixup item")
    context = data["context"]
    instruction = data["input"]
    if not isinstance(context, str):
        context = str(context)
    if not isinstance(instruction, str):
        instruction = str(instruction)
    return f"{context}\n\n{instruction}"


def should_load_as_multifieldqa_en_mixup(
    data: list, dataset_path: Path, original_dataset_name: str
) -> bool:
    if not data or not isinstance(data[0], dict):
        return False
    row0 = data[0]
    if not isinstance(row0.get("input"), str) or not isinstance(
        row0.get("context"), str
    ):
        return False
    alias = original_dataset_name.lower()
    aliases = (
        "multifieldqa_en_20k_40k",
        "multifieldqa_en_mixup",
        "multifieldqa_en_mixup_32k_ctx_20k_40k",
        "lic_mixup",
        "lic_mixup_32k_ctx_20k_40k",
        "hotpotwikiqa_mixup",
        "hotpotwikiqa_mixup_16k_ctx_20k_40k",
    )
    if alias in aliases:
        return True
    if Path(original_dataset_name).stem.lower() in aliases:
        return True
    return is_multifieldqa_en_mixup_dataset_path(dataset_path)


def load_multifieldqa_en_mixup_json_records(
    data: list, dataset_path: Path
) -> list[dict]:
    if not isinstance(data, list):
        raise ValueError(f"{dataset_path} must contain a JSON list")
    instances: list[dict] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(
                f"{dataset_path} index {index}: expected object, got {type(item)}"
            )
        instances.append({"turns": [format_multifieldqa_en_mixup_prompt(item)]})
    return instances


def is_swe_bench_style_json(
    data: list, dataset_path: Path, original_dataset_name: str
) -> bool:
    """SWE-bench Parquet export: list of dicts with string ``text`` (and usually ``instance_id``)."""
    if not data or not isinstance(data[0], dict):
        return False
    row0 = data[0]
    if "text" not in row0 or not isinstance(row0.get("text"), str):
        return False
    stem = dataset_path.stem.lower()
    alias_stem = Path(str(original_dataset_name)).stem.lower()
    explicit_name = stem.startswith("swe_bench") or alias_stem.startswith("swe_bench")
    return explicit_name or "instance_id" in row0


def load_swe_bench_json_instances(data: list, dataset_path: Path) -> list[dict]:
    instances: list[dict] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(
                f"{dataset_path} index {index}: expected object, got {type(item)}"
            )
        if "text" not in item or item["text"] is None:
            raise ValueError(f"Missing 'text' in {dataset_path} at index {index}")
        text = item["text"]
        if not isinstance(text, str):
            text = str(text)
        instances.append({"turns": [text]})
    return instances


def resolve_dataset_path(dataset_name: str) -> str:
    if not DATASET_PATH_FILE.is_file():
        return dataset_name

    with DATASET_PATH_FILE.open("r", encoding="utf-8") as f:
        dataset_paths = json.load(f)

    if not isinstance(dataset_paths, dict):
        raise ValueError(f"{DATASET_PATH_FILE} must contain a JSON object")

    return dataset_paths.get(dataset_name, dataset_name)


def default_specbench_question_jsonl() -> Path:
    """Repo-root ``Spec-Bench/data/spec_bench/question.jsonl``."""
    return (
        Path(__file__).resolve().parents[2]
        / "Spec-Bench/data/spec_bench/question.jsonl"
    )


def specbench_dataset_meta(dataset_alias: str) -> tuple[bool, str | None]:
    """``specbench`` → all categories; ``specbench_math`` → filter ``math``."""
    s = dataset_alias.strip().lower()
    if s == "specbench":
        return True, None
    if s.startswith("specbench_"):
        cat = s[len("specbench_") :].strip("_")
        return True, (cat if cat else None)
    return False, None


def parse_specbench_category(dataset_alias: str) -> str | None:
    is_sb, cat = specbench_dataset_meta(dataset_alias)
    return cat if is_sb else None


def expand_specbench_turn_string(turn: str) -> list[str]:
    """Sub-questions in one JSON turn: ``|||`` or comma-separated when each segment ends with ``?``."""
    turn = turn.strip()
    if not turn:
        return []
    if "|||" in turn:
        return [p.strip() for p in turn.split("|||") if p.strip()]
    if ", " in turn:
        raw_parts = [p.strip() for p in turn.split(", ") if p.strip()]
        if len(raw_parts) >= 2 and all(p.endswith("?") for p in raw_parts):
            return raw_parts
    return [turn]


def flatten_specbench_turns(turns: list) -> list[str]:
    expanded_turns: list[str] = []
    for turn in turns:
        if not isinstance(turn, str):
            turn = str(turn)
        expanded_turns.extend(expand_specbench_turn_string(turn))
    return expanded_turns


def load_specbench_question_jsonl(
    dataset_path: Path, original_dataset_name: str
) -> list[dict]:
    is_sb, category = specbench_dataset_meta(original_dataset_name)
    if not is_sb:
        raise ValueError(
            "internal: load_specbench_question_jsonl requires a specbench dataset name"
        )
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Spec-Bench file not found: {dataset_path}")
    instances: list[dict] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if (
                category is not None
                and str(obj.get("category", "")).lower() != category.lower()
            ):
                continue
            turns = obj.get("turns")
            if not isinstance(turns, list) or not turns:
                raise ValueError(
                    f"{dataset_path}:{line_number}: missing non-empty turns"
                )
            flat = flatten_specbench_turns(turns)
            if not flat:
                continue
            instances.append(
                {
                    "turns": flat,
                    "specbench_chain_turns": True,
                    "category": str(obj.get("category", "unknown")),
                    "question_id": obj.get("question_id"),
                }
            )
    if not instances:
        hint = f" for category={category!r}" if category else " (all categories)"
        raise ValueError(f"No Spec-Bench samples{hint} in {dataset_path}")
    return instances


def load_benchmark_dataset(dataset_name: str):
    original_dataset_name = dataset_name
    is_specbench, _ = specbench_dataset_meta(original_dataset_name)
    if is_specbench:
        resolved = resolve_dataset_path(original_dataset_name)
        sb_file = Path(resolved)
        if not sb_file.is_file() or sb_file.suffix != ".jsonl":
            sb_file = default_specbench_question_jsonl()
        if not sb_file.is_file():
            raise FileNotFoundError(
                f"Spec-Bench question.jsonl not found (tried {resolved} and {sb_file})."
            )
        return load_specbench_question_jsonl(sb_file, original_dataset_name)

    dataset_name = resolve_dataset_path(original_dataset_name)
    dataset_path = Path(dataset_name)
    if dataset_path.is_dir():
        if (
            original_dataset_name.lower() == "longbench_v2"
            or is_longbench_v2_dataset_path(dataset_path)
            or original_dataset_name.lower().startswith("longbench_v2_")
        ):
            return load_longbench_v2_shard_directory(
                dataset_path, original_dataset_name
            )
        raise ValueError(f"Unsupported dataset directory: {dataset_path}")

    if dataset_path.is_file() and dataset_path.suffix == ".json":
        with dataset_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{dataset_path} must contain a JSON list")

        if (
            original_dataset_name.lower() == "longbench_v2"
            or is_longbench_v2_dataset_path(dataset_path)
            or original_dataset_name.lower().startswith("longbench_v2_")
        ):
            return load_longbench_v2_json_records(
                data, dataset_path, original_dataset_name
            )

        if should_load_as_multifieldqa_en_mixup(
            data, dataset_path, original_dataset_name
        ):
            return load_multifieldqa_en_mixup_json_records(data, dataset_path)

        if is_swe_bench_style_json(data, dataset_path, original_dataset_name):
            return load_swe_bench_json_instances(data, dataset_path)

        raise ValueError(f"Unsupported JSON dataset: {dataset_path}")

    if dataset_path.is_file() and dataset_path.suffix == ".jsonl":
        instances = []
        with dataset_path.open("r", encoding="utf-8") as f:
            # Some exports use a ``.jsonl`` name but store one pretty-printed JSON array.
            head = f.read(8192)
            f.seek(0)
            if head.lstrip("\ufeff").lstrip().startswith("["):
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError(f"{dataset_path} must contain a JSON list")
                if (
                    original_dataset_name.lower() == "longbench_v2"
                    or is_longbench_v2_dataset_path(dataset_path)
                    or original_dataset_name.lower().startswith("longbench_v2_")
                ):
                    return load_longbench_v2_json_records(
                        data, dataset_path, original_dataset_name
                    )
                if should_load_as_multifieldqa_en_mixup(
                    data, dataset_path, original_dataset_name
                ):
                    return load_multifieldqa_en_mixup_json_records(data, dataset_path)
                if is_swe_bench_style_json(data, dataset_path, original_dataset_name):
                    return load_swe_bench_json_instances(data, dataset_path)
                raise ValueError(
                    f"Unsupported JSON-array dataset with .jsonl extension: {dataset_path}"
                )

            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                if "input" not in data:
                    raise ValueError(
                        f"Missing 'input' field in {dataset_path} at line {line_number}"
                    )
                if "context" not in data:
                    raise ValueError(
                        f"Missing 'context' field in {dataset_path} at line {line_number}"
                    )
                prompt = f"{data['context']}\nQuestion: {data['input']}"
                instances.append({"turns": [prompt]})
        return instances

    return load_and_process_dataset(dataset_name)


def select_max_samples(dataset, max_samples: int | None):
    if max_samples is None or len(dataset) <= max_samples:
        return dataset

    if hasattr(dataset, "shuffle") and hasattr(dataset, "select"):
        return dataset.shuffle(seed=0).select(range(max_samples))

    indices = list(range(len(dataset)))
    rng = random.Random(0)
    rng.shuffle(indices)
    return [dataset[i] for i in indices[:max_samples]]


def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def decode_weight(run) -> int:
    """Token count used for amortized decode metrics."""
    n = int(getattr(run, "num_tokens_for_decode_rate", run.num_output_tokens))
    return int(run.batch_size) * max(n, 1)


def decode_wall_seconds(run) -> float:
    return float(run.time_per_output_token) * decode_weight(run)


def compute_benchmark_stats(
    responses: list, block_size: int, verify_block_size: int
) -> dict:
    """Aggregate decode speedup and acceptance length over benchmark turn responses."""
    if not responses:
        return {
            "num_turns": 0,
            "token_weighted_speedup": 0.0,
            "throughput_ratio": 0.0,
            "baseline_s_per_token": 0.0,
            "dlite_s_per_token": 0.0,
            "unweighted_speedup": 0.0,
            "avg_accept_length": 0.0,
            "histogram": [0.0] * (verify_block_size + 1),
        }

    w1 = sum(decode_weight(r[1]) for r in responses)
    wb = sum(decode_weight(r[block_size]) for r in responses)
    d1 = sum(decode_wall_seconds(r[1]) for r in responses)
    db = sum(decode_wall_seconds(r[block_size]) for r in responses)
    t1 = d1 / max(w1, 1)
    tb = db / max(wb, 1)
    throughput1 = w1 / max(d1, 1e-30)
    throughputb = wb / max(db, 1e-30)
    t1_unweighted = float(np.mean([r[1].time_per_output_token for r in responses]))
    tb_unweighted = float(
        np.mean([r[block_size].time_per_output_token for r in responses])
    )

    acceptance_lengths = list(
        chain(*[r[block_size].acceptance_lengths for r in responses])
    )
    if acceptance_lengths:
        histogram = [
            acceptance_lengths.count(b) / len(acceptance_lengths)
            for b in range(verify_block_size + 1)
        ]
        avg_accept = float(sum(index * prob for index, prob in enumerate(histogram)))
    else:
        histogram = [0.0] * (verify_block_size + 1)
        avg_accept = 0.0

    return {
        "num_turns": len(responses),
        "token_weighted_speedup": t1 / max(tb, 1e-30),
        "throughput_ratio": throughputb / max(throughput1, 1e-30),
        "baseline_s_per_token": t1,
        "dlite_s_per_token": tb,
        "unweighted_speedup": t1_unweighted / max(tb_unweighted, 1e-30),
        "avg_accept_length": avg_accept,
        "histogram": histogram,
    }


def group_responses_by_category(responses: list) -> dict[str, list]:
    groups: dict[str, list] = defaultdict(list)
    for response in responses:
        groups[str(response.get("category", "unknown"))].append(response)
    return dict(groups)


def print_benchmark_stats(stats: dict, block_size: int, title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"  turns: {stats['num_turns']}")
    print(
        f"  token-weighted speedup: {stats['token_weighted_speedup']:.2f}x | "
        f"throughput ratio: {stats['throughput_ratio']:.2f}x | "
        f"unweighted speedup: {stats['unweighted_speedup']:.2f}x"
    )
    print(
        f"  decode s/token baseline={stats['baseline_s_per_token']:.6f} "
        f"dlite={stats['dlite_s_per_token']:.6f}"
    )
    histogram = stats["histogram"]
    hist_text = ", ".join(f"{x * 100:.1f}%" for x in histogram)
    print(f"  acceptance length histogram: [{hist_text}]")
    print(f"  average acceptance length: {stats['avg_accept_length']:.2f}")


@torch.inference_mode()
def run_benchmark_warmup(
    target: AutoModelForCausalLM,
    draft_model: DLiteDraftModel,
    tokenizer: AutoTokenizer,
    verify_block_size: int,
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    stop_token_ids: list[int],
    stochastic_verification_mode: str,
    compile_serial_head: bool,
) -> None:
    """One short baseline + speculative pass to warm CUDA kernels before timing."""
    warmup_prompt = [{"role": "user", "content": "Warmup."}]
    input_text = tokenizer.apply_chat_template(
        warmup_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    if batch_size > 1:
        input_ids = input_ids.expand(batch_size, -1).contiguous()
    warmup_new_tokens = min(16, max_new_tokens)
    target_generate(
        target=target,
        input_ids=input_ids,
        max_new_tokens=warmup_new_tokens,
        stop_token_ids=stop_token_ids,
        temperature=temperature,
        decode_timing_after_first_token=False,
    )
    dlite_generate(
        model=draft_model,
        target=target,
        input_ids=input_ids,
        max_new_tokens=warmup_new_tokens,
        verify_block_size=verify_block_size,
        stop_token_ids=stop_token_ids,
        temperature=temperature,
        decode_timing_after_first_token=False,
        stochastic_verification_mode=stochastic_verification_mode,
        compile_serial_head=compile_serial_head,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cuda_time()


@torch.inference_mode()
def target_generate(
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    decode_timing_after_first_token: bool = False,
) -> SimpleNamespace:
    batch_size_dim = input_ids.shape[0]
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    output_ids = torch.empty(
        (batch_size_dim, max_length), dtype=torch.long, device=input_ids.device
    )
    output_ids[:, :num_input_tokens] = input_ids
    position_ids = (
        torch.arange(max_length, device=input_ids.device)
        .unsqueeze(0)
        .expand(batch_size_dim, -1)
    )
    past_key_values_target = make_target_cache(target, max_length)
    stop_tensor = (
        torch.tensor(stop_token_ids, device=input_ids.device, dtype=torch.long)
        if stop_token_ids
        else None
    )

    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=False,
    )
    next_token = sample(output.logits, temperature)
    time_to_first_token = (cuda_time() - prefill_start) / batch_size_dim

    decode_start: float | None = (
        None if decode_timing_after_first_token else cuda_time()
    )
    start = input_ids.shape[1]
    while start < max_length:
        output_ids[:, start : start + 1] = next_token
        start += 1

        if decode_timing_after_first_token and decode_start is None:
            decode_start = cuda_time()

        if stop_tensor is not None and torch.all(
            torch.isin(next_token.squeeze(-1), stop_tensor)
        ):
            break
        if start >= max_length:
            break

        token_position_ids = position_ids[:, start - 1 : start]
        output = target(
            next_token,
            position_ids=token_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=False,
        )
        next_token = sample(output.logits, temperature)

    output_ids = output_ids[:, :start]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    if decode_start is None:
        decode_start = cuda_time()
    total_decode_time = cuda_time() - decode_start
    rate_tokens = max(
        num_output_tokens - (1 if decode_timing_after_first_token else 0), 1
    )
    time_per_output_token = total_decode_time / (batch_size_dim * rate_tokens)
    throughput_tokens_per_sec = (batch_size_dim * rate_tokens) / max(
        total_decode_time, 1e-9
    )

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        num_tokens_for_decode_rate=rate_tokens,
        batch_size=batch_size_dim,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        throughput_tokens_per_sec=float(throughput_tokens_per_sec),
        decode_wall_time=total_decode_time,
        acceptance_lengths=[1] * num_output_tokens,
    )


@torch.inference_mode()
def dlite_generate(
    model: DLiteDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    verify_block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    decode_timing_after_first_token: bool = False,
    stochastic_verification_mode: str = "match",
    compile_serial_head: bool = False,
) -> SimpleNamespace:
    target_cache = make_target_cache(
        target, input_ids.shape[1] + max_new_tokens + verify_block_size
    )
    output_ids = model.spec_generate(
        target=target,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids,
        temperature=temperature,
        decode_timing_after_first_token=decode_timing_after_first_token,
        verify_block_size=verify_block_size,
        stochastic_verification_mode=stochastic_verification_mode,
        compile_serial_head=compile_serial_head,
        target_cache=target_cache,
    )

    stats = model.get_last_decode_stats()
    bsz = int(input_ids.shape[0])
    num_input_tokens = input_ids.shape[1]
    num_output_tokens = output_ids.shape[1] - num_input_tokens
    decode_wall_time = float(stats.get("decode_wall_time", 0.0))
    rate_tokens = max(
        num_output_tokens - (1 if decode_timing_after_first_token else 0), 1
    )
    time_per_output_token = decode_wall_time / (bsz * max(rate_tokens, 1))
    throughput_tokens_per_sec = (bsz * rate_tokens) / max(decode_wall_time, 1e-9)

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        num_tokens_for_decode_rate=rate_tokens,
        batch_size=bsz,
        time_to_first_token=0.0,
        time_per_output_token=time_per_output_token,
        throughput_tokens_per_sec=float(throughput_tokens_per_sec),
        decode_wall_time=decode_wall_time,
        acceptance_lengths=stats.get("accept_lengths", []),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument(
        "--verify-block",
        "--verify_block",
        dest="verify_block",
        type=int,
        default=None,
        help="Verify only this many positions from the front of each drafted block "
        "(anchor-inclusive). block_size is the configured draft width. "
        "Default: verify the full block (proposal_length + 1).",
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--stochastic-verification-mode",
        choices=("match", "rejection"),
        default="match",
        help="Verification used when temperature > 0. 'match' preserves the "
        "existing exact-token validation; 'rejection' uses p/q acceptance and "
        "residual sampling. Temperature 0 always keeps greedy match validation.",
    )
    parser.add_argument(
        "--compile-serial-head",
        action="store_true",
        help="Compile the serial Sequential-head block sampler with torch.compile. "
        "Compilation happens during warmup.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Replicate each prompt along the batch dimension (expand) for throughput runs. "
        "Use temperature=0 so DLite speculative steps stay aligned across the batch.",
    )
    parser.add_argument(
        "--mask-token-id",
        type=int,
        default=None,
        help="Override mask token id (default: checkpoint dlite_config, then tokenizer).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading target/draft/tokenizer.",
    )
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")

    target, draft_model, tokenizer, draft_summary = load_dlite_benchmark_models(
        args, device
    )
    logger.info(
        "Target KV cache: preallocated for baseline and DLite; "
        "allocation during prefill, outside decode timing"
    )

    config_block_size = draft_model.block_size
    draft_block_len = draft_model.draft_block_len
    max_verify_block_size = draft_model.max_verify_block_size
    verify_block_size = (
        args.verify_block if args.verify_block is not None else max_verify_block_size
    )
    if not 1 <= verify_block_size <= max_verify_block_size:
        raise ValueError(
            f"--verify-block must be in [1, {max_verify_block_size}], "
            f"got {verify_block_size}"
        )
    logger.info(
        "Decode config: sequential_head={} sequential_rank={} "
        "model_role={} swa_window_size={} fuse_slots={} "
        "chs_num_layers={} condition_slots={}",
        draft_summary["sequential_head"],
        draft_summary["sequential_rank"],
        draft_summary["model_role"],
        draft_summary["swa_window_size"],
        draft_summary["fuse_slot_count"],
        draft_summary["chs_num_layers"],
        draft_summary["condition_slots"],
    )
    logger.info(
        "DLite decode: config_block_size={} draft_block_len={} "
        "verify_block_size={} verification_mode={} compile_serial_head={} "
        "(discarding {} draft proposals before target verification)",
        config_block_size,
        draft_block_len,
        verify_block_size,
        args.stochastic_verification_mode,
        args.compile_serial_head,
        draft_model.proposal_length - max(verify_block_size - 1, 0),
    )

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.batch_size > 1 and args.temperature > 1e-5:
        if args.stochastic_verification_mode == "rejection":
            raise ValueError(
                "--stochastic-verification-mode rejection with temperature>0 "
                "currently requires --batch-size 1."
            )
        logger.warning(
            "batch_size>1 with temperature>0 can desynchronize DLite across batch rows; "
            "prefer --temperature 0 for batched benchmarking."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    stop_token_ids = [
        token_id for token_id in [tokenizer.eos_token_id] if token_id is not None
    ]
    dataset = load_benchmark_dataset(args.dataset)
    dataset = select_max_samples(dataset, args.max_samples)

    if dist.is_main():
        print("Running CUDA warmup (baseline + DLite, short decode)...")
    run_benchmark_warmup(
        target=target,
        draft_model=draft_model,
        tokenizer=tokenizer,
        verify_block_size=verify_block_size,
        device=device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        stop_token_ids=stop_token_ids,
        stochastic_verification_mode=args.stochastic_verification_mode,
        compile_serial_head=args.compile_serial_head,
    )

    benchmark_start = cuda_time()
    responses = []
    indices = range(dist.rank(), len(dataset), dist.size())
    for idx in tqdm(indices, disable=not dist.is_main()):
        instance = dataset[idx]
        messages = []
        chain_turns = bool(instance.get("specbench_chain_turns"))
        decode_after_first = chain_turns
        sample_category = instance.get("category")
        prev_assistant = ""
        for turn_index, turn_q in enumerate(instance["turns"]):
            if chain_turns:
                user_content = (
                    turn_q if turn_index == 0 else f"{prev_assistant}\n\n{turn_q}"
                )
            else:
                user_content = turn_q
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(
                target.device
            )
            if args.batch_size > 1:
                input_ids = input_ids.expand(args.batch_size, -1).contiguous()
            category_suffix = (
                f" | category={sample_category}" if sample_category else ""
            )
            print(
                f"\n[Sample {idx} | Turn {turn_index}{category_suffix}] Input length: "
                f"{input_ids.shape[1]} tokens ({len(user_content)} chars), "
                f"batch_size={input_ids.shape[0]}"
            )

            response = {}
            response[1] = target_generate(
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=stop_token_ids,
                temperature=args.temperature,
                decode_timing_after_first_token=decode_after_first,
            )
            response[config_block_size] = dlite_generate(
                model=draft_model,
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                verify_block_size=verify_block_size,
                stop_token_ids=stop_token_ids,
                temperature=args.temperature,
                decode_timing_after_first_token=decode_after_first,
                stochastic_verification_mode=args.stochastic_verification_mode,
                compile_serial_head=args.compile_serial_head,
            )

            spec_response = response[config_block_size]
            generated_ids = spec_response.output_ids[
                0, spec_response.num_input_tokens :
            ]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            acceptance_lengths_text = ", ".join(
                [
                    f"{position}:{length}"
                    for position, length in enumerate(spec_response.acceptance_lengths)
                ]
            )
            avg_acceptance_length = np.mean(spec_response.acceptance_lengths)
            print(f"\n[Sample {idx} | Turn {turn_index}] Response:\n{output_text}")
            print(
                f"[Sample {idx} | Turn {turn_index}] Decode timing "
                f"(cuda wall after prefill, amortized s/token, batch={args.batch_size}): "
                f"baseline={response[1].time_per_output_token:.6f}, "
                f"dlite={spec_response.time_per_output_token:.6f} | "
                f"tok/s (batch total): baseline={response[1].throughput_tokens_per_sec:.2f}, "
                f"dlite={spec_response.throughput_tokens_per_sec:.2f}"
            )
            print(
                f"[Sample {idx} | Turn {turn_index}] Acceptance lengths (position:length): "
                f"{acceptance_lengths_text}"
            )
            print(
                f"[Sample {idx} | Turn {turn_index}] Average acceptance length: {avg_acceptance_length:.2f}"
            )

            messages.append({"role": "assistant", "content": output_text})
            if chain_turns:
                prev_assistant = output_text
            if sample_category is not None:
                response["category"] = sample_category
            if instance.get("question_id") is not None:
                response["question_id"] = instance["question_id"]
            responses.append(response)

    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))

    overall_stats = compute_benchmark_stats(
        responses, config_block_size, verify_block_size
    )
    print_benchmark_stats(
        overall_stats,
        config_block_size,
        title=f"Overall (batch_size={args.batch_size})",
    )

    category_groups = group_responses_by_category(responses)
    if len(category_groups) > 1 or (
        len(category_groups) == 1 and "unknown" not in category_groups
    ):
        print("\n=== Per-category breakdown ===")
        for category in sorted(category_groups):
            cat_stats = compute_benchmark_stats(
                category_groups[category], config_block_size, verify_block_size
            )
            print_benchmark_stats(
                cat_stats, config_block_size, title=f"category={category}"
            )

    total_elapsed_time = cuda_time() - benchmark_start
    print(f"Total elapsed time: {total_elapsed_time:.2f}s")


if __name__ == "__main__":
    main()
