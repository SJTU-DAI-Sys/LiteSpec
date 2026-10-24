"""Target/draft loading for DFlash-family benchmarks."""

from __future__ import annotations

import torch
from transformers import AutoConfig, AutoTokenizer

from litespec.modeling.config_utils import load_hf_target_model
from litespec.modeling.draft.registry import method_config, resolve_draft


def load_dflash_benchmark_models(args, device):
    config = AutoConfig.from_pretrained(args.draft_name_or_path)
    (architecture,) = config.architectures
    draft_class = resolve_draft(architecture)
    if draft_class.method_config_key != f"{args.algorithm}_config":
        raise ValueError(f"{args.algorithm} benchmark cannot load {architecture}")
    method_config(config)
    try:
        from flash_attn import flash_attn_func, flash_attn_varlen_func  # noqa: F401

        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
    target = (
        load_hf_target_model(
            args.model_name_or_path,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,
            trust_remote_code=args.trust_remote_code,
        )
        .to(device)
        .eval()
    )
    draft = (
        draft_class.from_pretrained(
            args.draft_name_or_path,
            config=config,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=args.trust_remote_code
    )
    method = method_config(draft.config)
    mask_id = args.mask_token_id
    if mask_id is None:
        mask_id = method.get("mask_token_id", tokenizer.mask_token_id)
    if mask_id is None:
        raise ValueError("DFlash-family benchmark requires an in-vocabulary mask token")
    draft.mask_token_id = int(mask_id)
    method["mask_token_id"] = int(mask_id)
    return target, draft, tokenizer


__all__ = ["load_dflash_benchmark_models"]
