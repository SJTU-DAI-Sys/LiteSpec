"""Shared training CLI definitions and validation of the effective arguments."""

from __future__ import annotations

import argparse
import math

from litespec.args import SGLangBackendArgs
from litespec.training.logging import TrackerArgs


def add_common_args(parser: argparse.ArgumentParser) -> None:
    model = parser.add_argument_group("model")
    model.add_argument("--target-model-path", required=True)
    model.add_argument("--target-model-backend", default="hf", choices=["hf", "sglang"])
    model.add_argument(
        "--embedding-key",
        default=None,
        help=(
            "Target checkpoint embedding key used by standalone/offline components. "
            "By default it is inferred from the checkpoint index."
        ),
    )
    model.add_argument(
        "--lm-head-key",
        default="lm_head.weight",
        help="Target checkpoint LM-head key used by standalone/offline components.",
    )
    model.add_argument(
        "--attention-backend",
        choices=("eager", "sdpa", "flex_attention"),
        default="flex_attention",
    )
    model.add_argument(
        "--flex-attention-backend",
        default="TRITON",
        type=str.upper,
        choices=["TRITON", "FLASH"],
        help="Draft-backbone FlexAttention kernel. FLASH pads Q/KV to full tiles.",
    )
    model.add_argument("--block-size", type=int, default=8)
    model.add_argument("--num-draft-layers", type=int, default=5)
    model.add_argument(
        "--target-layer-ids",
        help=(
            "Comma-separated zero-based target layer IDs. When provided, this "
            "takes precedence over --chs-num-layers."
        ),
    )
    model.add_argument(
        "--mask-token-id",
        type=int,
        default=None,
        help=(
            "Optional in-vocabulary MASK embedding row. By default, preserve the "
            "checkpoint value, use tokenizer.mask_token_id, or select the first "
            "model vocabulary row unused by the tokenizer."
        ),
    )
    model.add_argument("--num-anchors", type=int, default=512)
    model.add_argument(
        "--trust-remote-code", action=argparse.BooleanOptionalAction, default=False
    )

    data = parser.add_argument_group("dataset")
    data.add_argument(
        "--train-data-path",
        help="Online token-only JSONL. Mutually exclusive with --train-hidden-states-path.",
    )
    data.add_argument(
        "--train-hidden-states-path",
        help="Offline regen_full cache directory. Mutually exclusive with --train-data-path.",
    )
    data.add_argument(
        "--verify-hidden-cache-checksums",
        action="store_true",
        help="Independently read every cache file to verify SHA256 before training; disabled by default.",
    )
    data.add_argument("--chat-template", default="qwen")
    data.add_argument(
        "--is-preformatted", action=argparse.BooleanOptionalAction, default=False
    )
    data.add_argument(
        "--pad-to-max-length",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pad every microbatch to --max-length (benchmark/control mode).",
    )
    data.add_argument("--max-length", type=int, default=4096)
    data.add_argument("--batch-size", type=int, default=1)
    data.add_argument("--dataloader-num-workers", type=int, default=8)
    data.add_argument("--build-dataset-num-proc", type=int, default=8)
    data.add_argument("--cache-dir", default="./cache/train")

    train = parser.add_argument_group("training")
    train.add_argument("--accumulation-steps", type=int, default=1)
    train.add_argument("--max-grad-norm", type=float, default=1.0)
    train.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="AdamW weight decay when Muon is disabled; ignored with --use-muon.",
    )
    train.add_argument(
        "--use-muon", action=argparse.BooleanOptionalAction, default=False
    )
    train.add_argument("--muon-lr", type=float, default=0.02)
    train.add_argument("--muon-momentum", type=float, default=0.95)
    train.add_argument("--muon-weight-decay", type=float, default=0.1)
    train.add_argument(
        "--muon-matrices-per-rank",
        type=int,
        default=1,
        help="Maximum same-shape matrices per compute rank in each Muon batch.",
    )
    train.add_argument("--muon-adamw-lr", type=float, default=0.0004)
    train.add_argument(
        "--muon-low-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable Muon batch prefetch to further reduce temporary memory.",
    )
    train.add_argument("--muon-adamw-weight-decay", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--tp-size", type=int, default=1)
    train.add_argument(
        "--shard-draft-by-tp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run target prefill on the full TP-group batch, then train each "
            "TP rank on its corresponding batch slice."
        ),
    )
    train.add_argument("--dist-timeout", type=int, default=1200)
    train.add_argument("--resume-from")
    disagg = parser.add_argument_group("disaggregated target/draft execution")
    disagg.add_argument(
        "--disaggregate", action=argparse.BooleanOptionalAction, default=False
    )
    disagg.add_argument("--target-ranks-per-node", type=int)
    disagg.add_argument("--draft-ranks-per-node", type=int)
    disagg.add_argument("--target-tp-size", type=int)
    disagg.add_argument("--node-batch-size", type=int)
    disagg.add_argument("--draft-micro-batch-size", type=int)
    disagg.add_argument("--pipeline-depth", type=int, default=2)
    disagg.add_argument(
        "--profile", action=argparse.BooleanOptionalAction, default=False
    )
    sglang = parser.add_argument_group("sglang target backend")
    SGLangBackendArgs.add_args(sglang)

    output = parser.add_argument_group("output")
    output.add_argument("--output-dir", required=True)
    output.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help=(
            "Log and compute diagnostics every N optimizer steps. Metrics "
            "cover the complete logged gradient-accumulation window, including "
            "scheduled final and forced partial-window tail logs."
        ),
    )
    output.add_argument("--save-interval", type=int, default=20000)
    TrackerArgs.add_args(parser.add_argument_group("tracker"))


def validate_common_args(parser: argparse.ArgumentParser, args) -> None:
    """Reject invalid launch values before distributed/model initialization."""
    from litespec.modeling.draft.flex_attention import flex_kernel_options
    from litespec.training.muon import muon_config_from_args

    try:
        muon_config_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    if not args.use_muon and (
        not math.isfinite(args.weight_decay) or args.weight_decay < 0
    ):
        parser.error("--weight-decay must be finite and non-negative")

    if args.attention_backend == "flex_attention":
        try:
            flex_kernel_options(args.flex_attention_backend)
        except (ValueError, RuntimeError) as exc:
            parser.error(str(exc))
    positive_integer_args = (
        "block_size",
        "num_draft_layers",
        "num_anchors",
        "max_length",
        "batch_size",
        "build_dataset_num_proc",
        "accumulation_steps",
        "tp_size",
        "dist_timeout",
        "log_interval",
        "save_interval",
        "sglang_ep_size",
    )
    for name in positive_integer_args:
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if int(args.block_size) < 2:
        parser.error("--block-size must be at least 2")
    if int(args.max_length) < 2 * int(args.block_size):
        parser.error("--max-length must be at least twice --block-size")
    for name in ("dataloader_num_workers",):
        if int(getattr(args, name)) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if float(args.max_grad_norm) <= 0:
        parser.error("--max-grad-norm must be positive")
    if args.mask_token_id is not None and int(args.mask_token_id) < 0:
        parser.error("--mask-token-id must be non-negative")
    if bool(args.train_data_path) == bool(args.train_hidden_states_path):
        parser.error(
            "exactly one of --train-data-path and --train-hidden-states-path is required"
        )
    if args.train_hidden_states_path:
        if int(args.tp_size) != 1:
            parser.error("offline regen_full training requires --tp-size 1")
        if args.shard_draft_by_tp:
            parser.error(
                "offline regen_full training does not support --shard-draft-by-tp"
            )
    if args.shard_draft_by_tp:
        if (
            args.target_model_backend != "sglang"
            or args.tp_size <= 1
            or args.batch_size != args.tp_size
        ):
            parser.error(
                "--shard-draft-by-tp requires sglang, tp_size > 1, and batch_size == tp_size"
            )
    if args.disaggregate:
        import os

        local_world_size = os.environ.get("LOCAL_WORLD_SIZE")
        if (
            local_world_size
            and args.target_ranks_per_node is not None
            and args.draft_ranks_per_node is not None
        ):
            if args.target_ranks_per_node + args.draft_ranks_per_node != int(
                local_world_size
            ):
                parser.error(
                    "disaggregate target/draft rank total must equal nproc_per_node"
                )
        required_arg_names = (
            "target_ranks_per_node",
            "draft_ranks_per_node",
            "node_batch_size",
        )
        missing_arg_names = [
            name for name in required_arg_names if getattr(args, name) is None
        ]
        if missing_arg_names:
            parser.error(
                "--disaggregate requires "
                + ", ".join(f"--{name.replace('_', '-')}" for name in missing_arg_names)
            )
        if args.train_hidden_states_path:
            parser.error("--disaggregate supports --train-data-path only")
        if args.shard_draft_by_tp:
            parser.error("--disaggregate is incompatible with --shard-draft-by-tp")
        if args.target_tp_size is None:
            args.target_tp_size = args.tp_size
        for name in (
            "target_ranks_per_node",
            "draft_ranks_per_node",
            "target_tp_size",
            "node_batch_size",
            "pipeline_depth",
        ):
            if int(getattr(args, name)) <= 0:
                parser.error(f"--{name.replace('_', '-')} must be positive")
        if int(args.pipeline_depth) < 2:
            parser.error("--pipeline-depth must be at least 2")
        if int(args.target_ranks_per_node) % int(args.target_tp_size):
            parser.error(
                "--target-ranks-per-node must be divisible by --target-tp-size"
            )
        producers = int(args.target_ranks_per_node) // int(args.target_tp_size)
        if int(args.node_batch_size) % producers:
            parser.error("--node-batch-size must be divisible by target producers")
        if int(args.node_batch_size) % int(args.draft_ranks_per_node):
            parser.error("--node-batch-size must be divisible by draft ranks")
        if int(args.target_tp_size) % int(args.sglang_ep_size):
            parser.error("--target-tp-size must be divisible by --sglang-ep-size")
        if (
            int(args.target_tp_size) > 1 or int(args.sglang_ep_size) > 1
        ) and args.target_model_backend != "sglang":
            parser.error(
                "disaggregated TP/EP > 1 requires --target-model-backend sglang"
            )
        local_draft_batch = int(args.node_batch_size) // int(args.draft_ranks_per_node)
        if args.draft_micro_batch_size is None:
            args.draft_micro_batch_size = local_draft_batch
        if int(args.draft_micro_batch_size) <= 0 or (
            local_draft_batch % int(args.draft_micro_batch_size)
        ):
            parser.error(
                "--draft-micro-batch-size must divide the per-rank draft batch"
            )
        args.target_batch_size = int(args.node_batch_size) // producers
