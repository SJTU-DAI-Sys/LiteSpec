import json
import os
import shutil
import time
import torch
from typing import Optional
from pathlib import Path
from datasets import Dataset, load_dataset, load_from_disk, Features, Sequence, Value
from huggingface_hub import hf_hub_download


NEMOTRON_LOCAL_DATA_DIR = Path(
    os.environ.get("NEMOTRON_LOCAL_DATA_DIR", "datasets/nemotron")
)

SWE_BENCH_LONG_TEST_FILE = Path(
    os.environ.get("SWE_BENCH_LONG_TEST_FILE", "datasets/swe-bench/test.parquet")
)

LONGBENCH_V2_FILE = Path(
    os.environ.get("LONGBENCH_V2_FILE", "datasets/longbench-v2/data.json")
)

DATASET_CACHE_ROOT = Path(os.environ.get("DLITE_DATASET_CACHE", ".cache/datasets"))
DATASET_CACHE_VERSION = "v1"

def _load_livecodebench_jsonl_dataset():
    """Download LiveCodeBench lite JSONL shards via HF Hub (datasets no longer runs loading scripts)."""
    repo_id = "livecodebench/code_generation_lite"
    allowed_files = [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    ]
    local_paths = [
        hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        for filename in allowed_files
    ]
    return load_dataset("json", data_files={"test": local_paths})["test"]


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [(num_target_layers // 2)]
    start = 1
    end = num_target_layers - 3
    span = end - start
    target_layer_ids = [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]
    return target_layer_ids

def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    offset = 1
    selected_states = []
    for layer_id in layer_ids:
        selected_states.append(hidden_states[layer_id + offset])
    target_hidden = torch.cat(selected_states, dim=-1)
    return target_hidden

def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)

def _dataset_cache_dir(data_key: str) -> Path:
    safe_key = str(data_key).replace("/", "_")
    return DATASET_CACHE_ROOT / f"{safe_key}_{DATASET_CACHE_VERSION}"


def _wait_for_ready_file(ready_file: Path, timeout_seconds: int = 7200, poll_seconds: float = 1.0):
    deadline = time.time() + timeout_seconds
    while True:
        if ready_file.exists():
            return
        if time.time() >= deadline:
            raise TimeoutError(f"等待缓存数据准备超时: {ready_file}")
        time.sleep(max(0.2, poll_seconds))


def _write_ready_file(ready_file: Path, data_key: str):
    payload = {
        "dataset": data_key,
        "cache_version": DATASET_CACHE_VERSION,
        "timestamp": int(time.time()),
        "pid": os.getpid(),
    }
    with open(ready_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def _load_nemotron_parquet(parquet_files: list[str]):
    try:
        return load_dataset("parquet", data_files={"train": parquet_files})["train"]
    except ValueError as exc:
        error_msg = str(exc)
        if "Feature type 'List' not found" not in error_msg:
            raise

        # 兼容旧版 datasets 对 parquet metadata 中 List 类型的解析限制。
        # 回退为显式 features 读取，避免从 metadata 反序列化失败。
        print(
            "==> 检测到 parquet metadata 的 List 特征与当前 datasets 版本不兼容，"
            "改用显式 features 回退加载"
        )
        fallback_features = Features(
            {
                "messages": Sequence(
                    {
                        "role": Value("string"),
                        "content": Value("string"),
                    }
                )
            }
        )
        return load_dataset(
            "parquet",
            data_files={"train": parquet_files},
            split="train",
            features=fallback_features,
        )


def _build_and_process_dataset(data_key: str):

    # Math datasets
    if data_key == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        prompt_fmt = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
    
    elif data_key == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
    
    elif data_key == "aime24":
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_key == "aime25":
        dataset = load_dataset("MathArena/aime_2025", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_key == "apex25":
        dataset = load_dataset("MathArena/apex_2025", split="train")
        prompt_fmt = "{problem}"
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
        
    elif data_key == "arxivmath":
        dataset = load_dataset("MathArena/arxivmath-0226", split="train")
        prompt_fmt = "{problem}"
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    # Chat datasets 
    elif data_key == "alpaca":
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        dataset = dataset.map(lambda x: {"formatted_input": (f"{x['instruction']}\n\nInput:\n{x['input']}" if x['input'] else x['instruction'])})
        dataset = dataset.map(lambda x: {"turns": [x["formatted_input"]]})

    elif data_key == "mt-bench":
        dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        dataset = dataset.map(lambda x: {"turns": x["prompt"]})

    elif data_key == "longbench_v2":
        if not LONGBENCH_V2_FILE.exists():
            raise FileNotFoundError(f"LongBench_v2 json file not found: {LONGBENCH_V2_FILE}")

        with LONGBENCH_V2_FILE.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError(f"{LONGBENCH_V2_FILE} must contain a JSON list")

        def format_longbench_v2(doc):
            return {
                "turns": [f"{doc['context']}\n\nQuestion: {doc['question']}"],
                "answer": str(doc.get("answer", "")),
            }

        target_features = Features(
            {
                "turns": Sequence(Value("large_string")),
                "answer": Value("string"),
            }
        )
        dataset = Dataset.from_list(rows).map(
            format_longbench_v2,
            remove_columns=list(rows[0].keys()) if rows else None,
            features=target_features,
        )

    elif data_key == "nemotron":
        if not NEMOTRON_LOCAL_DATA_DIR.exists():
            raise FileNotFoundError(
                f"Nemotron local parquet directory not found: {NEMOTRON_LOCAL_DATA_DIR}"
            )

        parquet_files = sorted(str(path) for path in NEMOTRON_LOCAL_DATA_DIR.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found under Nemotron directory: {NEMOTRON_LOCAL_DATA_DIR}"
            )

        dataset = _load_nemotron_parquet(parquet_files)

        def format_nemotron_prompt(example):
            messages = example.get("messages") or []
            user_messages = [
                message.get("content", "")
                for message in messages
                if message.get("role") == "user" and message.get("content")
            ]
            prompt = user_messages[0] if user_messages else ""
            task_value = (
                example.get("task_type")
                or example.get("task")
                or example.get("category")
                or example.get("source")
                or "unknown"
            )
            return {"turns": [prompt], "task_type": str(task_value)}

        target_features = Features(
            {
                "turns": Sequence(Value("large_string")),
                "task_type": Value("string"),
            }
        )
        dataset = dataset.map(
            format_nemotron_prompt,
            remove_columns=dataset.column_names,
            features=target_features,
        )

    # Coding datasets
    elif data_key == "humaneval":
        dataset = load_dataset("openai/openai_humaneval", split="test")
        prompt_fmt = "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```"
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
    
    elif data_key == "mbpp":
        dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        dataset = dataset.map(lambda x: {"turns": [x["prompt"]]})
    
    elif data_key == "lbpp":
        LBPP_PY_TEST_URL = "https://huggingface.co/datasets/CohereLabs/lbpp/resolve/main/python/test.parquet"
        dataset = load_dataset("parquet", data_files={"test": LBPP_PY_TEST_URL})["test"]
        dataset = dataset.map(lambda x: {"turns": [x["instruction"]]})

    elif data_key == "swe-bench":
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        prompt_fmt = "Problem Statement:\n{problem_statement}\nPlease fix the issue described above."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_key == "swe-bench-long":
        if not SWE_BENCH_LONG_TEST_FILE.exists():
            raise FileNotFoundError(
                f"SWE-bench long parquet file not found: {SWE_BENCH_LONG_TEST_FILE}"
            )

        dataset = load_dataset(
            "parquet",
            data_files={"test": str(SWE_BENCH_LONG_TEST_FILE)},
            split="test",
        )

        target_features = Features({"turns": Sequence(Value("large_string"))})
        dataset = dataset.map(
            lambda x: {"turns": [x["text"]]},
            remove_columns=dataset.column_names,
            features=target_features,
        )
    
    elif data_key == "livecodebench":
        dataset = _load_livecodebench_jsonl_dataset()
        def format_lcb(doc):
            system_prompt = (
                "You are an expert Python programmer. You will be given a question (problem specification) "
                "and will generate a correct Python program that matches the specification and passes all tests. "
                "You will NOT return anything except for the program"
            )
            question_block = f"### Question:\n{doc['question_content']}"
            if doc.get("starter_code"):
                format_message = "### Format: Use the following code structure:"
                code_block = f"```python\n{doc['starter_code']}\n```"
            else:
                format_message = "### Format: Write your code in the following format:"
                code_block = "```python\n# YOUR CODE HERE\n```"
            answer_footer = "### Answer: (use the provided format with backticks)"
            return f"{system_prompt}\n\n{question_block}\n\n{format_message}\n{code_block}\n\n{answer_footer}"
        target_features = Features({"turns": Sequence(Value("large_string"))})
        dataset = dataset.map(
            lambda x: {"turns": [format_lcb(x)]},
            remove_columns=dataset.column_names,
            features=target_features
        )

    else:
        raise ValueError(f"Unsupported dataset: {data_key}")
    
    return dataset


def load_and_process_dataset(data_name: str):
    data_key = data_name.lower()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    cache_dir = _dataset_cache_dir(data_key)
    ready_file = cache_dir / "_READY.json"

    # nemotron 已预处理完成时，直接读本地缓存目录，不再走 parquet split/map。
    if data_key == "nemotron":
        if cache_dir.exists() and (ready_file.exists() or (cache_dir / "dataset_info.json").exists()):
            print(f"==> 直接加载已处理数据集: {cache_dir}")
            return load_from_disk(str(cache_dir))
        raise FileNotFoundError(
            "未找到 nemotron 的已处理数据集缓存目录。"
            f"期望路径: {cache_dir}. "
            "请先准备好该目录后再运行。"
        )

    if cache_dir.exists() and ready_file.exists():
        print(f"==> 复用处理后数据集缓存: {cache_dir}")
        return load_from_disk(str(cache_dir))

    if world_size > 1 and rank != 0:
        print(f"==> [rank {rank}] 等待主进程准备缓存数据集: {cache_dir}")
        _wait_for_ready_file(ready_file)
        return load_from_disk(str(cache_dir))

    dataset = _build_and_process_dataset(data_key)

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_cache_dir = cache_dir.parent / f".{cache_dir.name}.tmp.{os.getpid()}"
    if tmp_cache_dir.exists():
        shutil.rmtree(tmp_cache_dir)

    dataset.save_to_disk(str(tmp_cache_dir))

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    os.replace(str(tmp_cache_dir), str(cache_dir))
    _write_ready_file(ready_file, data_key)

    print(f"==> 已保存处理后数据集缓存: {cache_dir}")
    return dataset

def build_hybrid_attention_mask(
    bsz: int,
    prefix_len: int,
    block_size: int,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Build a hybrid attention mask combining:
    - Causal attention for the prefix (prefix_len): each token can only see previous tokens
    - Bidirectional attention within last block_size (generated block): tokens can see each other and can see all prefix tokens

    Args:
        bsz: Batch size
        prefix_len: Length of prefix sequence with causal attention
        block_size: Length of generated block with bidirectional attention
        device: Device to create the mask on

    Returns:
        attn_mask: [B, 1, T_total, T_total] bool, where T_total = prefix_len + block_size
                   True = can attend (visible), False = cannot attend (masked)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    T_total = prefix_len + block_size
    
    # Initialize attention mask to False (all masked by default)
    attn_mask = torch.zeros(
        (bsz, 1, T_total, T_total), dtype=torch.bool, device=device
    )

    # 1. Within prefix: causal mask (each token can only see previous tokens including itself)
    #    [0:prefix_len, 0:prefix_len]
    pos_q = torch.arange(prefix_len, device=device).view(1, prefix_len, 1)
    pos_k = torch.arange(prefix_len, device=device).view(1, 1, prefix_len)
    causal_mask = pos_k <= pos_q  # [1, prefix_len, prefix_len], True = visible
    attn_mask[:, 0, :prefix_len, :prefix_len] = causal_mask

    # 2. Block tokens looking at prefix: all prefix tokens are visible
    #    [prefix_len:T_total, 0:prefix_len]
    attn_mask[:, 0, prefix_len:, :prefix_len] = True

    # 3. Within block: bidirectional (all tokens can see each other)
    #    [prefix_len:T_total, prefix_len:T_total]
    attn_mask[:, 0, prefix_len:, prefix_len:] = True

    return attn_mask
