"""Shared DLite target/draft loading for benchmark and profiling."""

from __future__ import annotations

import argparse
from typing import Any

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from litespec.modeling.draft.dlite import DLiteDraftModel


def resolve_mask_token_id(
    draft_model: DLiteDraftModel,
    tokenizer: AutoTokenizer,
    *,
    cli_mask_token_id: int | None = None,
) -> int:
    """Resolve mask token id using the same priority as training."""
    if cli_mask_token_id is not None:
        return int(cli_mask_token_id)

    mask_token_id = draft_model.mask_token_id
    if mask_token_id is not None:
        return int(mask_token_id)

    if tokenizer.mask_token_id is not None:
        return int(tokenizer.mask_token_id)

    tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
    if tokenizer.mask_token_id is None:
        raise ValueError(
            "mask_token_id is None. Pass --mask-token-id, use a checkpoint with "
            "dlite_config['mask_token_id'], or a tokenizer with mask_token_id."
        )
    return int(tokenizer.mask_token_id)


def dlite_config_summary(draft_model: DLiteDraftModel) -> dict[str, Any]:
    return {
        "architecture_version": draft_model.architecture_version,
        "model_role": draft_model.model_role,
        "swa_window_size": draft_model.swa_window_size,
        "fuse_slot_count": draft_model.fuse_slot_count,
        "chs_num_layers": draft_model.chs_num_layers,
        "condition_slots": draft_model.condition_slot_count,
        "target_layer_ids": draft_model.target_layer_ids,
        "history_layer_ids": draft_model.history_layer_ids,
        "block_size": int(draft_model.block_size),
        "sequential_head": draft_model.sequential_head_type,
        "sequential_rank": draft_model.sequential_rank,
        "mask_token_id": draft_model.mask_token_id,
    }


def log_dlite_config(draft_model: DLiteDraftModel) -> dict[str, Any]:
    summary = dlite_config_summary(draft_model)
    logger.info(
        "DLite draft: architecture_version={} model_role={} swa_window_size={} "
        "fuse_slots={} chs_num_layers={} condition_slots={} "
        "target_layer_ids={} history_layer_ids={} block_size={} sequential_head={} "
        "sequential_rank={} mask_token_id={}",
        summary["architecture_version"],
        summary["model_role"],
        summary["swa_window_size"],
        summary["fuse_slot_count"],
        summary["chs_num_layers"],
        summary["condition_slots"],
        summary["target_layer_ids"],
        summary["history_layer_ids"],
        summary["block_size"],
        summary["sequential_head"],
        summary["sequential_rank"],
        summary["mask_token_id"],
    )
    return summary


def validate_decode_config(draft_model: DLiteDraftModel) -> None:
    """Log serial-head inference settings from the loaded checkpoint."""
    summary = dlite_config_summary(draft_model)
    sequential_head = str(summary["sequential_head"])
    query_layout = "Q=[embed(a), MASK...]"
    rnn_init = "RNN zero-initialized"
    if draft_model.is_student:
        logger.info(
            "PivotQ student: {}, local RoPE; CHS is context KV; {}. "
            "block_size={} proposals={} query_len={}",
            query_layout,
            rnn_init,
            summary["block_size"],
            draft_model.proposal_length,
            draft_model.draft_query_length,
        )
    else:
        logger.info(
            "SWA teacher: KV=[fuse(a-W)..fuse(a-2), CHS(a-1)], "
            "{}, global RoPE; {}. "
            "block_size={} proposals={} W={}",
            query_layout,
            rnn_init,
            summary["block_size"],
            draft_model.proposal_length,
            summary["swa_window_size"],
        )

    logger.info(
        "Sequential head enabled for inference: type={} rank={}",
        sequential_head,
        summary["sequential_rank"],
    )
    logger.info("Draft logits come directly from the sequential head.")


def has_flash_attention() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        logger.warning(
            "flash_attn is not installed. Falling back to torch.sdpa. "
            "The speedup will be lower."
        )
        return False


def load_dlite_benchmark_models(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[AutoModelForCausalLM, DLiteDraftModel, AutoTokenizer, dict[str, Any]]:
    installed_flash_attn = has_flash_attention()
    attn_impl = "flash_attention_2" if installed_flash_attn else "sdpa"

    target = (
        AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,
            trust_remote_code=getattr(args, "trust_remote_code", False),
        )
        .to(device)
        .eval()
    )

    draft_model = (
        DLiteDraftModel.from_pretrained(
            args.draft_name_or_path,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,
            trust_remote_code=getattr(args, "trust_remote_code", False),
        )
        .to(device)
        .eval()
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=getattr(args, "trust_remote_code", False),
    )
    mask_token_id = resolve_mask_token_id(
        draft_model,
        tokenizer,
        cli_mask_token_id=getattr(args, "mask_token_id", None),
    )
    draft_model.mask_token_id = mask_token_id
    if draft_model.config.dlite_config is None:
        draft_model.config.dlite_config = {}
    draft_model.config.dlite_config["mask_token_id"] = mask_token_id
    logger.info("Using mask_token_id={}", mask_token_id)

    summary = log_dlite_config(draft_model)
    validate_decode_config(draft_model)
    return target, draft_model, tokenizer, summary
