"""Quality and throughput benchmark for DFlash, DFlash2, and DSpark."""

from __future__ import annotations

import argparse
import random
from itertools import chain

import numpy as np
import torch
from rich import print
from tqdm import tqdm

from evaluation import distributed as dist
from evaluation.benchmark import load_benchmark_dataset, select_max_samples, target_generate
from evaluation.dflash_model_loading import load_dflash_benchmark_models
from litespec.modeling.kv_cache import make_target_cache


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=("dflash", "dflash2", "dspark"), required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--draft-name-or-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mask-token-id", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    if args.batch_size != 1:
        parser.error("DFlash-family benchmark currently requires --batch-size 1")
    return args


def main():
    args = parse_args()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")
    target, draft, tokenizer = load_dflash_benchmark_models(args, device)
    if dist.is_main():
        print(
            "Target KV cache: preallocated for baseline and draft verification; "
            "allocation during prefill, outside decode timing; draft KV: dynamic"
        )
    stop_ids = [value for value in (tokenizer.eos_token_id,) if value is not None]
    dataset = select_max_samples(load_benchmark_dataset(args.dataset), args.max_samples)
    responses = []
    for index in tqdm(range(dist.rank(), len(dataset), dist.size()), disable=not dist.is_main()):
        messages = []
        for turn in dataset[index]["turns"]:
            messages.append({"role": "user", "content": str(turn)})
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
            baseline = target_generate(
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=stop_ids,
                temperature=args.temperature,
            )
            output_ids = draft.spec_generate(
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=stop_ids,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                target_cache=make_target_cache(
                    target, input_ids.shape[1] + args.max_new_tokens + draft.block_size
                ),
            )
            stats = draft.get_last_decode_stats()
            generated = output_ids[0, input_ids.shape[1] :]
            output_text = tokenizer.decode(generated, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            accept = list(stats.get("accept_lengths", []))
            wall = float(stats.get("decode_wall_time", 0.0))
            count = int(generated.numel())
            responses.append(
                {
                    "tokens": count,
                    "wall": wall,
                    "baseline_tps": baseline.throughput_tokens_per_sec,
                    "accept": accept,
                    "text": output_text,
                }
            )
            print(
                f"[{args.algorithm} sample={index}] tokens={count} "
                f"tok/s={count / max(wall, 1e-9):.2f} "
                f"avg_accept={float(np.mean(accept)) if accept else 0.0:.2f}"
            )
    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))
    total_tokens = sum(item["tokens"] for item in responses)
    total_wall = sum(item["wall"] for item in responses)
    accepted = list(chain.from_iterable(item["accept"] for item in responses))
    print(
        {
            "algorithm": args.algorithm,
            "samples": len(responses),
            "tokens": total_tokens,
            "tokens_per_second": total_tokens / max(total_wall, 1e-9),
            "mean_acceptance_length": float(np.mean(accepted)) if accepted else 0.0,
        }
    )


if __name__ == "__main__":
    main()
