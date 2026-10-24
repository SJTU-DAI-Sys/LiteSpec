"""DLite SFT, teacher and two-stage preparation and objectives."""

from __future__ import annotations

import argparse
import copy
import gc
import math
import os
from types import SimpleNamespace
from typing import Optional

import torch
import torch.distributed as dist

from litespec.core.dlite import (
    OnlineDLiteModel,
    compute_stage1_distillation_loss,
    gather_target_prefill_logits,
    prepare_history_hidden_states,
    prepare_target_hidden,
)
from litespec.core.outputs import RatioTerms, TrainingOutput
from litespec.distributed.transport import DLiteBatchPacket, DLitePacketSpec
from litespec.modeling.config_utils import load_text_model_config
from litespec.modeling.draft.dlite import (
    DLITE_ARCHITECTURE_VERSION,
    DLiteDraftModel,
    build_target_layer_ids,
)
from litespec.training.algorithms import Phase, TrainingAdapter, get_training_adapter
from litespec.training.config import add_common_args, validate_common_args
from litespec.training.logging import get_tracker_class
from litespec.training.setup import (
    build_train_dataloader,
    hidden_states_to_cuda,
    load_cached_target_data,
    project_cached_target_logits,
    required_hidden_layer_ids,
    select_tp_rank_batch,
)

TWO_STAGE_PARAMETER_ORDER = "stage1_then_stage2"


def two_stage_optimizer_parameters(student):
    """Register backbone parameters first, then the initially frozen head."""
    student.requires_grad_(True)
    head_ids = (
        {id(parameter) for parameter in student.sequential_head.parameters()}
        if student.sequential_head is not None
        else set()
    )
    parameters = list(student.parameters())
    return [p for p in parameters if id(p) not in head_ids] + [
        p for p in parameters if id(p) in head_ids
    ]


def add_dlite_args(parser, *, mode):
    model = parser.add_argument_group("DLite architecture")
    model.add_argument("--chs-num-layers", type=int, default=7)
    model.add_argument("--sequential-head", default="rnn", choices=["rnn"])
    model.add_argument("--sequential-rank", type=int, default=256)
    if mode == "teacher":
        model.add_argument("--swa-window-size", type=int, default=32)


def validate_dlite_args(parser, args):
    if args.attention_backend != "flex_attention":
        parser.error("DLite training requires attention_backend=flex_attention")
    for name in ("chs_num_layers", "sequential_rank"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.mode == "teacher" and args.swa_window_size <= 0:
        parser.error("swa_window_size must be positive")


def build_draft_config(args, *, model_role: str, source_config=None):
    config = (
        load_text_model_config(
            args.target_model_path, trust_remote_code=args.trust_remote_code
        )
        if source_config is None
        else source_config
    )
    if source_config is None:
        target_depth = int(config.num_hidden_layers)
        config.num_hidden_layers = int(args.num_draft_layers)
        config.num_target_layers = target_depth
        config.block_size = int(args.block_size)
    if args.target_layer_ids:
        target_layer_ids = [
            int(value.strip())
            for value in str(args.target_layer_ids).split(",")
            if value.strip()
        ]
        if not target_layer_ids:
            raise ValueError("--target-layer-ids must contain at least one layer ID.")
        if target_layer_ids != sorted(set(target_layer_ids)):
            raise ValueError(
                "--target-layer-ids must be unique and in strictly increasing order."
            )
        invalid = [
            layer_id
            for layer_id in target_layer_ids
            if not 0 <= layer_id < int(config.num_target_layers)
        ]
        if invalid:
            raise ValueError(
                f"Target layer IDs {invalid} are outside [0, "
                f"{int(config.num_target_layers) - 1}]."
            )
        chs_num_layers = len(target_layer_ids)
        args.chs_num_layers = chs_num_layers
    else:
        chs_num_layers = int(args.chs_num_layers)
        target_layer_ids = build_target_layer_ids(
            int(config.num_target_layers), chs_num_layers
        )
    dlite = dict(
        architecture_version=DLITE_ARCHITECTURE_VERSION,
        model_role=model_role,
        chs_num_layers=chs_num_layers,
        target_layer_ids=target_layer_ids,
        sequential_head=args.sequential_head,
        sequential_rank=int(args.sequential_rank),
        mask_token_id=(None if args.mask_token_id is None else int(args.mask_token_id)),
    )
    if model_role == "swa_teacher":
        dlite["swa_window_size"] = int(args.swa_window_size)
        dlite["history_layer_ids"] = [
            0,
            int(config.num_target_layers) // 2,
            int(config.num_target_layers) - 1,
        ]
    config.dlite_config = dlite
    config.architectures = ["DLiteDraftModel"]
    config._attn_implementation = args.attention_backend
    layer_types = list(getattr(config, "layer_types", []) or [])
    config.layer_types = (
        layer_types[: config.num_hidden_layers]
        if len(layer_types) >= config.num_hidden_layers
        else ["full_attention"] * config.num_hidden_layers
    )
    return config


def build_draft_model(args, *, model_role: str, source_config=None):
    from litespec.training.algorithms import get_training_adapter

    return get_training_adapter("dlite").build_model(
        args, model_role=model_role, source_config=source_config
    )


def parse_sft_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Direct supervised DLite student training"
    )
    add_common_args(parser)
    add_dlite_args(parser, mode="sft")
    parser.add_argument(
        "--local-position",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the student local-position layout (required).",
    )
    parser.add_argument(
        "--init-from",
        help=(
            "Initialize student weights/config from a checkpoint while starting "
            "a fresh optimizer, scheduler, and data cursor."
        ),
    )
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.04)
    parser.add_argument("--final-ce-weight", type=float, default=0.1)
    parser.add_argument("--l1-loss-weight", type=float, default=1.0)
    parser.add_argument("--base-lm-ce-weight", type=float, default=0.0)
    parser.add_argument("--loss-decay-gamma", type=float)
    parser.add_argument("--base-lm-ce-decay-gamma", type=float)
    args = parser.parse_args(argv)
    args.algorithm = "dlite"
    args.mode = "sft"

    validate_common_args(parser, args)
    validate_dlite_args(parser, args)
    if args.resume_from and args.init_from:
        parser.error("--resume-from and --init-from are mutually exclusive")
    if not args.local_position:
        parser.error("Direct DLite student training requires --local-position")
    if args.num_epochs <= 0:
        parser.error("--num-epochs must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if not 0.0 <= args.warmup_ratio <= 1.0:
        parser.error("--warmup-ratio must be in [0, 1]")
    loss_weights = (
        args.final_ce_weight,
        args.l1_loss_weight,
        args.base_lm_ce_weight,
    )
    if any(weight < 0 for weight in loss_weights):
        parser.error("Loss weights must be non-negative")
    if sum(loss_weights) == 0:
        parser.error("At least one loss weight must be positive")
    get_tracker_class(args.report_to).validate_args(parser, args)
    return args


def parse_teacher_args(argv=None):
    parser = argparse.ArgumentParser(description="Train DLite SWA teacher")
    add_common_args(parser)
    add_dlite_args(parser, mode="teacher")
    parser.add_argument(
        "--init-from",
        help=(
            "Initialize draft weights/config from a checkpoint but start a fresh "
            "optimizer, data cursor, and global step."
        ),
    )
    parser.add_argument("--num-epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.04)
    parser.add_argument("--final-ce-weight", type=float, default=1.0)
    parser.add_argument("--l1-loss-weight", type=float, default=1.0)
    parser.add_argument("--base-lm-ce-weight", type=float, default=0.0)
    parser.add_argument("--loss-decay-gamma", type=float)
    parser.add_argument("--base-lm-ce-decay-gamma", type=float)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional short-run limit, primarily for reproducible benchmarks.",
    )
    parser.add_argument(
        "--final-save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the final checkpoint; --no-final-save skips it for benchmarks.",
    )
    args = parser.parse_args(argv)
    args.algorithm = "dlite"
    args.mode = "teacher"
    validate_common_args(parser, args)
    validate_dlite_args(parser, args)
    if args.resume_from and args.init_from:
        parser.error("--resume-from and --init-from are mutually exclusive")
    if args.num_epochs <= 0:
        parser.error("--num-epochs must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if not 0.0 <= args.warmup_ratio <= 1.0:
        parser.error("--warmup-ratio must be in [0, 1]")
    loss_weights = (
        args.final_ce_weight,
        args.l1_loss_weight,
        args.base_lm_ce_weight,
    )
    if any(weight < 0 for weight in loss_weights):
        parser.error("Teacher loss weights must be non-negative")
    if sum(loss_weights) == 0:
        parser.error("At least one teacher loss weight must be positive")
    get_tracker_class(args.report_to).validate_args(parser, args)
    return args


def parse_two_stage_args(argv=None):
    parser = argparse.ArgumentParser(description="Two-stage DLite distillation")
    add_common_args(parser)
    add_dlite_args(parser, mode="two_stage")
    parser.add_argument("--teacher-draft-path")
    parser.add_argument("--stage1-epochs", type=int, required=True)
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Base LR for the single optimizer/scheduler shared by both stages.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        help="Warmup ratio over the combined Stage1+Stage2 optimizer steps.",
    )
    parser.add_argument("--stage1-kl-weight", type=float, default=1.0)
    parser.add_argument("--stage1-loss-decay-gamma", type=float)
    parser.add_argument("--stage2-epochs", type=int, required=True)
    parser.add_argument("--stage2-final-ce-weight", type=float, default=1.0)
    parser.add_argument("--stage2-l1-weight", type=float, default=1.0)
    parser.add_argument("--stage2-base-ce-weight", type=float, default=0.0)
    parser.add_argument("--stage2-loss-decay-gamma", type=float)
    parser.add_argument("--stage2-base-ce-decay-gamma", type=float)
    args = parser.parse_args(argv)
    args.algorithm = "dlite"
    args.mode = "two_stage"
    validate_common_args(parser, args)
    validate_dlite_args(parser, args)
    for name in ("stage2_epochs", "accumulation_steps"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.stage1_epochs <= 0:
        parser.error("--stage1-epochs must be positive")
    if args.learning_rate is None or args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.warmup_ratio is None or not 0.0 <= args.warmup_ratio <= 1.0:
        parser.error("--warmup-ratio must be in [0, 1]")
    if args.stage1_kl_weight <= 0:
        parser.error("--stage1-kl-weight must be positive")
    stage2_weights = (
        args.stage2_final_ce_weight,
        args.stage2_l1_weight,
        args.stage2_base_ce_weight,
    )
    if any(weight < 0 for weight in stage2_weights):
        parser.error("Stage 2 loss weights must be non-negative")
    if sum(stage2_weights) == 0:
        parser.error("At least one Stage 2 loss weight must be positive")
    if args.resume_from is None and args.teacher_draft_path is None:
        parser.error("--teacher-draft-path is required for fresh training")
    get_tracker_class(args.report_to).validate_args(parser, args)
    return args


def _set_student_stage1_trainable(student: DLiteDraftModel) -> None:
    student.requires_grad_(True)
    if student.sequential_head is not None:
        student.sequential_head.requires_grad_(False)


def _set_student_stage2_trainable(student: DLiteDraftModel) -> None:
    student.requires_grad_(True)


def _cosine_transition_scales(
    batch_idx: int, num_batches: int, *, disaggregated=False
) -> tuple[float, float]:
    """Fade Stage 1 out and Stage 2 in over one transition epoch."""
    if num_batches <= 0:
        raise ValueError(f"Transition dataloader must be non-empty, got {num_batches}")
    if not 0 <= int(batch_idx) < int(num_batches):
        raise ValueError(
            f"Transition batch index must be in [0, {num_batches}), got {batch_idx}"
        )
    if disaggregated:
        progress = float(batch_idx) / max(num_batches - 1, 1)
        stage1_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return stage1_scale, 1.0 - stage1_scale
    if num_batches == 1:
        return 0.5, 0.5
    progress = float(batch_idx) / float(num_batches - 1)
    stage2_scale = 0.5 * (1.0 - math.cos(math.pi * progress))
    return 1.0 - stage2_scale, stage2_scale


def _copy_serial_head(teacher: DLiteDraftModel, student: DLiteDraftModel) -> None:
    teacher_signature = (
        teacher.architecture_version,
        teacher.sequential_head_type,
        teacher.sequential_rank,
        teacher.block_size,
        teacher.config.vocab_size,
    )
    student_signature = (
        student.architecture_version,
        student.sequential_head_type,
        student.sequential_rank,
        student.block_size,
        student.config.vocab_size,
    )
    if teacher_signature != student_signature:
        raise ValueError(
            f"Teacher/student serial configuration mismatch: {teacher_signature} != {student_signature}"
        )
    if teacher.sequential_head is None or student.sequential_head is None:
        if teacher.sequential_head is not student.sequential_head:
            raise ValueError("Teacher/student serial heads do not match")
        return
    student.sequential_head.load_state_dict(
        teacher.sequential_head.state_dict(), strict=True
    )


def validate_resume_model(state: Optional[dict], draft: DLiteDraftModel) -> None:
    if state is None:
        return
    expected = {
        "architecture_version": draft.architecture_version,
        "model_role": draft.model_role,
        "target_layer_ids": [int(value) for value in draft.target_layer_ids],
        "num_target_layers": int(draft.config.num_target_layers),
        "target_hidden_size": int(draft.config.hidden_size),
    }
    for key, current in expected.items():
        saved = state[key]
        if saved != current:
            raise ValueError(
                f"Disaggregated resume model mismatch for {key}: "
                f"checkpoint={saved!r}, current={current!r}"
            )


def _sync_args_from_draft(args, draft: DLiteDraftModel) -> None:
    args.block_size = int(draft.block_size)
    args.num_draft_layers = int(draft.config.num_hidden_layers)
    args.chs_num_layers = int(draft.chs_num_layers)
    args.target_layer_ids = ",".join(str(value) for value in draft.target_layer_ids)
    args.sequential_head = draft.sequential_head_type
    args.sequential_rank = int(draft.sequential_rank)
    if draft.is_teacher:
        args.swa_window_size = int(draft.swa_window_size)


def _initialize_drafts(args, mode: str, common_state: Optional[dict]):
    resume_stage = common_state["training_stage"] if common_state else None
    if common_state is not None:
        allowed_stages = (
            {"stage1", "transition", "stage2"} if mode == "two_stage" else {mode}
        )
        if resume_stage not in allowed_stages:
            raise ValueError(f"Invalid {mode} checkpoint stage: {resume_stage!r}")
    teacher = None
    if mode == "teacher":
        source = args.resume_from or args.init_from
        draft = (
            DLiteDraftModel.from_pretrained(
                source,
                torch_dtype=torch.bfloat16,
                attn_implementation="flex_attention",
            ).cuda()
            if source
            else build_draft_model(args, model_role="swa_teacher")
        )
        _sync_args_from_draft(args, draft)
        return draft, None, resume_stage
    if mode == "sft":
        source = args.resume_from or args.init_from
        draft = (
            DLiteDraftModel.from_pretrained(
                source,
                torch_dtype=torch.bfloat16,
                attn_implementation="flex_attention",
            ).cuda()
            if source
            else build_draft_model(args, model_role="student")
        )
        if not draft.is_student:
            raise ValueError("SFT requires a student draft")
        _sync_args_from_draft(args, draft)
        return draft, None, resume_stage

    if common_state:
        student = DLiteDraftModel.from_pretrained(
            args.resume_from,
            torch_dtype=torch.bfloat16,
            attn_implementation="flex_attention",
        ).cuda()
        if not student.is_student:
            raise ValueError("two-stage resume requires a student checkpoint")
    else:
        teacher = (
            DLiteDraftModel.from_pretrained(
                args.teacher_draft_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flex_attention",
            )
            .cuda()
            .eval()
        )
        if not teacher.is_teacher:
            raise ValueError("--teacher-draft-path must contain an swa_teacher")
        teacher.requires_grad_(False)
        _sync_args_from_draft(args, teacher)
        student = build_draft_model(
            args,
            model_role="student",
            source_config=copy.deepcopy(teacher.config),
        )
        _copy_serial_head(teacher, student)
    _sync_args_from_draft(args, student)
    if resume_stage in ("stage1", "transition") and teacher is None:
        if not args.teacher_draft_path:
            raise ValueError("Stage 1/transition resume requires --teacher-draft-path")
        teacher = (
            DLiteDraftModel.from_pretrained(
                args.teacher_draft_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flex_attention",
            )
            .cuda()
            .eval()
        )
        teacher.requires_grad_(False)
    return student, teacher, resume_stage


def _draft_metadata(draft, teacher=None) -> dict:
    layer_ids = set(int(value) for value in draft.target_layer_ids)
    history_ids = set()
    if teacher is not None:
        layer_ids.update(int(value) for value in teacher.target_layer_ids)
        history_ids.update(int(value) for value in teacher.history_layer_ids)
    if draft.is_teacher:
        history_ids.update(int(value) for value in draft.history_layer_ids)
    return {
        "target_layer_ids": sorted(layer_ids),
        "history_layer_ids": sorted(history_ids),
        "num_target_layers": int(draft.config.num_target_layers),
        "hidden_size": int(draft.config.hidden_size),
        "block_size": int(draft.block_size),
        "draft_query_length": int(draft.draft_query_length),
        "chs_num_layers": int(draft.chs_num_layers),
    }


def aligned_anchor_count(num_anchors: int, query_length: int, chs_slots: int) -> int:
    alignment = 1
    for tile, value in ((256, query_length), (128, query_length + chs_slots)):
        alignment = math.lcm(alignment, tile // math.gcd(tile, int(value)))
    return math.ceil(int(num_anchors) / alignment) * alignment


def _sample_anchors(args, topology, loss_mask, batch_id, metadata):
    bsz, seq_len = loss_mask.shape
    count = int(args.num_anchors)
    if args.flex_attention_backend == "FLASH":
        count = aligned_anchor_count(
            count, metadata["draft_query_length"], metadata["chs_num_layers"]
        )
    if topology.is_target_leader:
        max_anchor = max(seq_len - int(args.block_size), 0)
        valid_anchor_mask = loss_mask[:, : max_anchor + 1] > 0.5
        valid_anchor_mask = valid_anchor_mask & (loss_mask[:, 1 : max_anchor + 2] > 0.5)
        generator = torch.Generator(device=loss_mask.device)
        generator.manual_seed(
            int(args.seed)
            + 1_000_003 * int(batch_id)
            + 10_007 * int(topology.node_rank)
            + 101 * int(topology.target_replica_local_rank)
        )
        random = torch.rand(
            valid_anchor_mask.shape, generator=generator, device=loss_mask.device
        )
        random.masked_fill_(~valid_anchor_mask, 2.0)
        width = min(count, valid_anchor_mask.size(1))
        indices = random.argsort(dim=1)[:, :width]
        selected = torch.gather(valid_anchor_mask, 1, indices)
        sentinel = max_anchor + 1
        indices.masked_fill_(~selected, sentinel)
        if width < count:
            indices = torch.cat(
                [indices, indices.new_full((bsz, count - width), sentinel)], dim=1
            )
        indices = indices.sort(dim=1).values
        block_keep_mask = indices.ne(sentinel)
        anchor_positions = torch.where(
            block_keep_mask, indices, torch.zeros_like(indices)
        )
        if not bool(block_keep_mask.any()):
            raise ValueError("batch has no valid DLite anchors")
    else:
        anchor_positions = torch.empty((bsz, count), dtype=torch.long, device="cuda")
        block_keep_mask = torch.empty((bsz, count), dtype=torch.bool, device="cuda")
    for value in (anchor_positions, block_keep_mask):
        dist.broadcast(
            value,
            src=topology.target_tp_leader_global_rank,
            group=topology.target_tp_group,
        )
    return anchor_positions, block_keep_mask


def _prediction_hidden(target_last_hidden, anchor_positions, block_size):
    offsets = torch.arange(block_size - 1, device=anchor_positions.device).view(
        1, 1, -1
    )
    positions = anchor_positions.unsqueeze(-1) + offsets
    expanded = target_last_hidden.unsqueeze(1).expand(
        -1, anchor_positions.size(1), -1, -1
    )
    return torch.gather(
        expanded,
        2,
        positions.unsqueeze(-1).expand(-1, -1, -1, target_last_hidden.size(-1)),
    )


def _produce_packet(args, topology, target, phase, metadata, values, batch_id):
    input_ids, attention_mask, loss_mask = values
    target_output = target.generate_training_data(
        input_ids, attention_mask, loss_mask, return_logits=False
    )
    anchor_positions, block_keep_mask = _sample_anchors(
        args, topology, loss_mask, batch_id, metadata
    )
    if not topology.is_target_leader:
        return None
    target_hidden_states = target_output.hidden_states
    target_hidden = prepare_target_hidden(
        target_hidden_states,
        anchor_positions,
        metadata["target_layer_ids"],
        metadata["num_target_layers"],
    ).to(torch.bfloat16)
    target_history_hidden = None
    if phase.need_history:
        target_history_hidden = prepare_history_hidden_states(
            target_hidden_states,
            metadata["history_layer_ids"],
            metadata["num_target_layers"],
        ).to(torch.bfloat16)
    target_prediction_hidden = None
    if phase.need_prediction_hidden:
        final_layer_id = metadata["num_target_layers"] - 1
        target_last_hidden = (
            target_hidden_states[final_layer_id]
            if isinstance(target_hidden_states, dict)
            else target_hidden_states[-1]
        )
        target_prediction_hidden = _prediction_hidden(
            target_last_hidden, anchor_positions, metadata["block_size"]
        ).to(torch.bfloat16)
    return DLiteBatchPacket(
        input_ids=input_ids,
        loss_mask=loss_mask,
        anchor_positions=anchor_positions,
        block_keep_mask=block_keep_mask,
        target_hidden=target_hidden,
        target_history_hidden=target_history_hidden,
        target_prediction_hidden=target_prediction_hidden,
    )


def _packet_spec(args, topology, phase, metadata):
    local_batch_size = (
        int(args.node_batch_size) // topology.target_replicas_per_node
        if topology.is_target
        else int(args.node_batch_size) // topology.draft_ranks_per_node
    )
    return DLitePacketSpec(
        batch_size=local_batch_size,
        max_length=int(args.max_length),
        num_anchors=(
            aligned_anchor_count(
                args.num_anchors,
                metadata["draft_query_length"],
                metadata["chs_num_layers"],
            )
            if args.flex_attention_backend == "FLASH"
            else int(args.num_anchors)
        ),
        num_target_layers=len(metadata["target_layer_ids"]),
        hidden_size=metadata["hidden_size"],
        prediction_length=metadata["block_size"] - 1,
        num_history_layers=(
            len(metadata["history_layer_ids"]) if phase.need_history else 0
        ),
        include_target_prediction_hidden=phase.need_prediction_hidden,
    )


def _select_packet_layers(packet, available_layer_ids, required_layer_ids):
    layer_indices = [
        available_layer_ids.index(int(layer_id)) for layer_id in required_layer_ids
    ]
    return packet.target_hidden[:, :, layer_indices, :]


def _target_logits(packet, phase, metadata, lm_head):
    with torch.no_grad():
        if packet.target_prediction_hidden is not None:
            return lm_head(packet.target_prediction_hidden)
        if packet.target_history_hidden is None:
            raise ValueError(
                f"phase {phase.name} has no final hidden for target logits"
            )
        final_layer = metadata["num_target_layers"] - 1
        history_index = metadata["history_layer_ids"].index(final_layer)
        return project_cached_target_logits(
            packet.target_history_hidden[:, :, history_index, :],
            packet.anchor_positions,
            metadata["block_size"],
            lm_head,
        )


def _make_online(args, draft, components, process_group, *, teacher_defaults=False):
    kwargs = dict(
        draft_model=draft,
        target_lm_head=components.lm_head,
        target_embed_tokens=components.embed_tokens,
        mask_token_id=args.mask_token_id,
        block_size=draft.block_size,
        attention_backend=args.attention_backend,
        flex_attention_backend=args.flex_attention_backend,
        num_anchors=args.num_anchors,
        process_group=process_group,
    )
    if not teacher_defaults:
        if hasattr(args, "stage2_final_ce_weight"):
            kwargs.update(
                loss_decay_gamma=args.stage2_loss_decay_gamma,
                final_ce_weight=args.stage2_final_ce_weight,
                l1_loss_weight=args.stage2_l1_weight,
                base_lm_ce_weight=args.stage2_base_ce_weight,
                base_lm_ce_decay_gamma=args.stage2_base_ce_decay_gamma,
                use_target_greedy_ce_labels=True,
            )
        else:
            kwargs.update(
                loss_decay_gamma=args.loss_decay_gamma,
                final_ce_weight=args.final_ce_weight,
                l1_loss_weight=args.l1_loss_weight,
                base_lm_ce_weight=args.base_lm_ce_weight,
                base_lm_ce_decay_gamma=args.base_lm_ce_decay_gamma,
                use_target_greedy_ce_labels=True,
            )
    return OnlineDLiteModel(**kwargs)


def _supervised_step(session, data, *, collect_detailed_metrics=True):
    args = session.args
    student = session.draft
    draft_online = session.draft_online
    target = session.target
    components = session.components
    fsdp_model = session.fsdp_model
    tp_draft_rank = session.tp_draft_rank
    input_ids = data["input_ids"].cuda(non_blocking=True)
    attention_mask = data["attention_mask"].cuda(non_blocking=True)
    loss_mask = data["loss_mask"].cuda(non_blocking=True)
    anchor_positions, block_keep_mask = draft_online.sample_anchor_positions(
        input_ids.size(1), loss_mask
    )
    if args.train_hidden_states_path:
        target_hidden_states, target_logits = load_cached_target_data(
            data,
            anchor_positions=anchor_positions,
            block_size=student.block_size,
            lm_head=components.lm_head,
            need_logits=True,
        )
    else:
        target_output = target.generate_training_data(
            input_ids, attention_mask, loss_mask
        )
        target_hidden_states = hidden_states_to_cuda(target_output.hidden_states)
        target_prefill_logits = target_output.logits.cuda()
    if tp_draft_rank is not None:
        input_ids = select_tp_rank_batch(input_ids, tp_draft_rank)
        loss_mask = select_tp_rank_batch(loss_mask, tp_draft_rank)
        anchor_positions = select_tp_rank_batch(anchor_positions, tp_draft_rank)
        block_keep_mask = select_tp_rank_batch(block_keep_mask, tp_draft_rank)
        target_hidden_states = select_tp_rank_batch(target_hidden_states, tp_draft_rank)
        target_prefill_logits = select_tp_rank_batch(
            target_prefill_logits, tp_draft_rank
        )
    if not args.train_hidden_states_path:
        target_logits = gather_target_prefill_logits(
            target_prefill_logits, anchor_positions, student.block_size
        )
        del target_output, target_prefill_logits
    output = get_training_adapter("dlite").training_step(
        fsdp_model,
        dict(
            input_ids=input_ids,
            hidden_states=target_hidden_states,
            loss_mask=loss_mask,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            target_prefill_logits=target_logits,
            target_logits_are_gathered=True,
            collect_detailed_metrics=collect_detailed_metrics,
        ),
    )
    return output


def _distillation_step(session, data, *, collect_detailed_metrics=True):
    args = session.args
    student = session.draft
    student_online = session.draft_online
    teacher_online = session.teacher_online
    target = session.target
    components = session.components
    fsdp_model = session.fsdp_model
    tp_draft_rank = session.tp_draft_rank
    input_ids = data["input_ids"].cuda(non_blocking=True)
    attention_mask = data["attention_mask"].cuda(non_blocking=True)
    loss_mask = data["loss_mask"].cuda(non_blocking=True)
    anchor_positions, block_keep_mask = student_online.sample_anchor_positions(
        input_ids.size(1), loss_mask
    )
    if args.train_hidden_states_path:
        target_hidden_states, _ = load_cached_target_data(
            data,
            anchor_positions=anchor_positions,
            block_size=student.block_size,
            lm_head=components.lm_head,
            need_logits=False,
        )
    else:
        target_output = target.generate_training_data(
            input_ids, attention_mask, loss_mask, return_logits=False
        )
        target_hidden_states = hidden_states_to_cuda(target_output.hidden_states)
        del target_output
    if tp_draft_rank is not None:
        # The TP target sees the shared full batch.  From this point on,
        # teacher and student on rank r both consume only sample r.
        input_ids = select_tp_rank_batch(input_ids, tp_draft_rank)
        loss_mask = select_tp_rank_batch(loss_mask, tp_draft_rank)
        anchor_positions = select_tp_rank_batch(anchor_positions, tp_draft_rank)
        block_keep_mask = select_tp_rank_batch(block_keep_mask, tp_draft_rank)
        target_hidden_states = select_tp_rank_batch(target_hidden_states, tp_draft_rank)
    student_batch = student_online.prepare_batch(
        input_ids,
        target_hidden_states,
        loss_mask,
        anchor_positions=anchor_positions,
        block_keep_mask=block_keep_mask,
    )
    with torch.no_grad():
        teacher_batch = teacher_online.prepare_batch(
            input_ids,
            target_hidden_states,
            loss_mask,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            shared_query_embeddings=student_batch.query_embeddings,
        )
        teacher_hidden = teacher_online.forward_backbone(
            teacher_batch, seq_len=input_ids.size(1)
        )
        teacher_serial_logits = teacher_online.compute_serial_logits(
            teacher_hidden, teacher_batch
        )
    student_hidden, student_serial_logits = get_training_adapter("dlite").training_step(
        fsdp_model,
        dict(
            prepared_batch=student_batch,
            seq_len=input_ids.size(1),
            return_backbone_and_serial_logits=True,
        ),
    )
    loss, _kl_loss, loss_numerator, loss_denominator = compute_stage1_distillation_loss(
        student_serial_logits=student_serial_logits,
        teacher_serial_logits=teacher_serial_logits,
        raw_weight_mask=student_batch.raw_weight_mask,
        kl_weight=args.stage1_kl_weight,
        loss_decay_gamma=args.stage1_loss_decay_gamma,
    )
    return TrainingOutput(
        loss=loss,
        loss_terms=RatioTerms(loss_numerator, loss_denominator),
        ratio_metrics=(
            {
                "kl": RatioTerms(
                    loss_numerator / args.stage1_kl_weight,
                    loss_denominator,
                )
            }
            if collect_detailed_metrics
            else {}
        ),
    )


def _transition_step(session, data, *, collect_detailed_metrics=True):
    args = session.args
    student = session.draft
    student_online = session.draft_online
    teacher_online = session.teacher_online
    target = session.target
    components = session.components
    fsdp_model = session.fsdp_model
    tp_draft_rank = session.tp_draft_rank
    stage1_scale, stage2_scale = _cosine_transition_scales(
        session.state.next_batch, session.steps_per_epoch
    )
    input_ids = data["input_ids"].cuda(non_blocking=True)
    attention_mask = data["attention_mask"].cuda(non_blocking=True)
    loss_mask = data["loss_mask"].cuda(non_blocking=True)
    anchor_positions, block_keep_mask = student_online.sample_anchor_positions(
        input_ids.size(1), loss_mask
    )
    if args.train_hidden_states_path:
        target_hidden_states, target_logits = load_cached_target_data(
            data,
            anchor_positions=anchor_positions,
            block_size=student.block_size,
            lm_head=components.lm_head,
            need_logits=True,
        )
    else:
        target_output = target.generate_training_data(
            input_ids, attention_mask, loss_mask
        )
        target_hidden_states = hidden_states_to_cuda(target_output.hidden_states)
        target_prefill_logits = target_output.logits.cuda()
    if tp_draft_rank is not None:
        input_ids = select_tp_rank_batch(input_ids, tp_draft_rank)
        loss_mask = select_tp_rank_batch(loss_mask, tp_draft_rank)
        anchor_positions = select_tp_rank_batch(anchor_positions, tp_draft_rank)
        block_keep_mask = select_tp_rank_batch(block_keep_mask, tp_draft_rank)
        target_hidden_states = select_tp_rank_batch(target_hidden_states, tp_draft_rank)
        target_prefill_logits = select_tp_rank_batch(
            target_prefill_logits, tp_draft_rank
        )
    if not args.train_hidden_states_path:
        target_logits = gather_target_prefill_logits(
            target_prefill_logits, anchor_positions, student.block_size
        )
        del target_output, target_prefill_logits
    student_batch = student_online.prepare_batch(
        input_ids,
        target_hidden_states,
        loss_mask,
        anchor_positions=anchor_positions,
        block_keep_mask=block_keep_mask,
    )
    with torch.no_grad():
        teacher_batch = teacher_online.prepare_batch(
            input_ids,
            target_hidden_states,
            loss_mask,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            shared_query_embeddings=student_batch.query_embeddings,
        )
        teacher_hidden = teacher_online.forward_backbone(
            teacher_batch, seq_len=input_ids.size(1)
        )
        teacher_serial_logits = teacher_online.compute_serial_logits(
            teacher_hidden, teacher_batch
        )
    (
        student_hidden,
        student_serial_logits,
        stage2_output,
    ) = get_training_adapter("dlite").training_step(
        fsdp_model,
        dict(
            prepared_batch=student_batch,
            seq_len=input_ids.size(1),
            target_prefill_logits=target_logits,
            target_logits_are_gathered=True,
            collect_detailed_metrics=collect_detailed_metrics,
            return_transition_outputs=True,
        ),
    )
    stage2_loss = stage2_output.loss
    stage2_loss_numerator = stage2_output.loss_terms.numerator
    stage2_loss_denominator = stage2_output.loss_terms.denominator

    (
        stage1_loss,
        _kl_loss,
        stage1_loss_numerator,
        stage1_loss_denominator,
    ) = compute_stage1_distillation_loss(
        student_serial_logits=student_serial_logits,
        teacher_serial_logits=teacher_serial_logits,
        raw_weight_mask=student_batch.raw_weight_mask,
        kl_weight=args.stage1_kl_weight,
        loss_decay_gamma=args.stage1_loss_decay_gamma,
    )
    loss = stage1_scale * stage1_loss + stage2_scale * stage2_loss
    loss_numerator = (
        stage1_scale * stage1_loss_numerator + stage2_scale * stage2_loss_numerator
    )
    loss_denominator = (
        stage1_scale * stage1_loss_denominator + stage2_scale * stage2_loss_denominator
    )
    zero_terms = (
        RatioTerms(stage2_loss.new_zeros(()), stage2_loss_denominator)
        if collect_detailed_metrics
        else None
    )
    return TrainingOutput(
        loss=loss,
        loss_terms=RatioTerms(loss_numerator, loss_denominator),
        ratio_metrics=(
            {
                "stage1_loss": RatioTerms(
                    stage1_loss_numerator, stage1_loss_denominator
                ),
                "stage2_loss": stage2_output.loss_terms,
                "kl": RatioTerms(
                    stage1_loss_numerator / args.stage1_kl_weight,
                    stage1_loss_denominator,
                ),
                "accuracy": stage2_output.ratio_metrics["hard_label/token_accuracy"],
                "hard_label/block_accepted_length": stage2_output.ratio_metrics[
                    "hard_label/block_accepted_length"
                ],
                "expected_accept_len": stage2_output.ratio_metrics[
                    "expected_accept_len"
                ],
                "final_ce": stage2_output.ratio_metrics["objective/final_ce"],
                "base_ce": stage2_output.ratio_metrics.get(
                    "objective/base_ce", zero_terms
                ),
                "l1_distance": stage2_output.ratio_metrics.get(
                    "objective/l1_distance", zero_terms
                ),
                "stage1_weight_scale": RatioTerms(
                    loss.new_tensor(stage1_scale), loss.new_ones(())
                ),
                "stage2_weight_scale": RatioTerms(
                    loss.new_tensor(stage2_scale), loss.new_ones(())
                ),
            }
            if collect_detailed_metrics
            else {}
        ),
    )


def _packet_step(
    session, packet, microbatch_start, microbatch_end, *, collect_detailed_metrics=True
):
    args = session.args
    mode = args.mode
    phase = session.phase
    metadata = session.packet_metadata
    draft, teacher = session.draft, session.teacher
    draft_online, teacher_online = session.draft_online, session.teacher_online
    components, fsdp_model = session.components, session.fsdp_model
    topology = session.topology
    batch_id = session.state.global_step
    phase_start = batch_id - session.state.phase_step
    phase_total = phase.batch_count(session.steps_per_epoch)
    input_ids = packet.input_ids[microbatch_start:microbatch_end]
    loss_mask = packet.loss_mask[microbatch_start:microbatch_end]
    anchor_positions = packet.anchor_positions[microbatch_start:microbatch_end]
    block_keep_mask = packet.block_keep_mask[microbatch_start:microbatch_end]
    target_hidden = _select_packet_layers(
        packet,
        metadata["target_layer_ids"],
        draft.target_layer_ids,
    )[microbatch_start:microbatch_end]
    prediction_hidden = (
        packet.target_prediction_hidden[microbatch_start:microbatch_end]
        if packet.target_prediction_hidden is not None
        else None
    )
    # Packet storage has a fixed transport capacity. TRITON itself
    # does not need tile padding, so remove the unused tail before
    # constructing the backbone inputs. FLASH keeps the aligned
    # physical shape selected by ``_packet_spec``.
    if args.flex_attention_backend == "TRITON":
        natural_width = int(block_keep_mask.sum(dim=1).max().item())
        if natural_width <= 0:
            raise ValueError("microbatch has no valid DLite anchors")
        anchor_positions = anchor_positions[:, :natural_width]
        block_keep_mask = block_keep_mask[:, :natural_width]
        target_hidden = target_hidden[:, :natural_width]
        if prediction_hidden is not None:
            prediction_hidden = prediction_hidden[:, :natural_width]
    history = (
        packet.target_history_hidden[microbatch_start:microbatch_end]
        if packet.target_history_hidden is not None
        else None
    )
    if mode == "two_stage" and phase.name in ("stage1", "transition"):
        if teacher_online is None:
            raise ValueError("Stage 1/transition requires the teacher draft")
        student_batch = draft_online.prepare_batch(
            input_ids,
            None,
            loss_mask,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            target_hidden=target_hidden,
        )
        teacher_hidden = _select_packet_layers(
            packet,
            metadata["target_layer_ids"],
            teacher.target_layer_ids,
        )[microbatch_start:microbatch_end]
        with torch.no_grad():
            teacher_batch = teacher_online.prepare_batch(
                input_ids,
                None,
                loss_mask,
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                shared_query_embeddings=student_batch.query_embeddings,
                target_hidden=teacher_hidden,
                raw_history_hidden=history,
            )
            teacher_prediction = teacher_online.forward_backbone(
                teacher_batch, seq_len=input_ids.size(1)
            )
            teacher_logits = teacher_online.compute_serial_logits(
                teacher_prediction, teacher_batch
            )
        if phase.name == "stage1":
            _student_hidden, student_logits = get_training_adapter(
                "dlite"
            ).training_step(
                fsdp_model,
                dict(
                    prepared_batch=student_batch,
                    seq_len=input_ids.size(1),
                    return_backbone_and_serial_logits=True,
                ),
            )
            loss, kl, numerator, denominator = compute_stage1_distillation_loss(
                student_serial_logits=student_logits,
                teacher_serial_logits=teacher_logits,
                raw_weight_mask=student_batch.raw_weight_mask,
                kl_weight=args.stage1_kl_weight,
                loss_decay_gamma=args.stage1_loss_decay_gamma,
                process_group=topology.draft_group,
            )
            metric_output = TrainingOutput(
                loss=loss,
                loss_terms=RatioTerms(numerator, denominator),
                ratio_metrics=(
                    {
                        "kl": RatioTerms(
                            numerator / args.stage1_kl_weight,
                            denominator,
                        )
                    }
                    if collect_detailed_metrics
                    else {}
                ),
            )
        else:
            target_logits = _target_logits(
                DLiteBatchPacket(
                    input_ids,
                    loss_mask,
                    anchor_positions,
                    block_keep_mask,
                    target_hidden,
                    history,
                    None,
                ),
                phase,
                metadata,
                components.lm_head,
            )
            _, student_logits, supervised = get_training_adapter("dlite").training_step(
                fsdp_model,
                dict(
                    prepared_batch=student_batch,
                    seq_len=input_ids.size(1),
                    target_prefill_logits=target_logits,
                    target_logits_are_gathered=True,
                    collect_detailed_metrics=collect_detailed_metrics,
                    return_transition_outputs=True,
                ),
            )
            stage2_loss = supervised.loss
            stage2_num = supervised.loss_terms.numerator
            stage2_den = supervised.loss_terms.denominator

            stage1_loss, kl, stage1_num, stage1_den = compute_stage1_distillation_loss(
                student_serial_logits=student_logits,
                teacher_serial_logits=teacher_logits,
                raw_weight_mask=student_batch.raw_weight_mask,
                kl_weight=args.stage1_kl_weight,
                loss_decay_gamma=args.stage1_loss_decay_gamma,
                process_group=topology.draft_group,
            )
            stage1_scale, stage2_scale = _cosine_transition_scales(
                batch_id - phase_start, phase_total, disaggregated=True
            )
            loss = stage1_scale * stage1_loss + stage2_scale * stage2_loss
            numerator = stage1_scale * stage1_num + stage2_scale * stage2_num
            denominator = stage1_scale * stage1_den + stage2_scale * stage2_den
            zero_terms = (
                RatioTerms(stage2_loss.new_zeros(()), stage2_den)
                if collect_detailed_metrics
                else None
            )
            metric_output = TrainingOutput(
                loss=loss,
                loss_terms=RatioTerms(numerator, denominator),
                ratio_metrics=(
                    {
                        "kl": RatioTerms(
                            stage1_num / args.stage1_kl_weight,
                            stage1_den,
                        ),
                        "stage2_loss": supervised.loss_terms,
                        "accuracy": supervised.ratio_metrics[
                            "hard_label/token_accuracy"
                        ],
                        "hard_label/block_accepted_length": supervised.ratio_metrics[
                            "hard_label/block_accepted_length"
                        ],
                        "expected_accept_len": supervised.ratio_metrics[
                            "expected_accept_len"
                        ],
                        "final_ce": supervised.ratio_metrics["objective/final_ce"],
                        "base_ce": supervised.ratio_metrics.get(
                            "objective/base_ce", zero_terms
                        ),
                        "l1_distance": supervised.ratio_metrics.get(
                            "objective/l1_distance", zero_terms
                        ),
                    }
                    if collect_detailed_metrics
                    else {}
                ),
            )
    else:
        target_logits = _target_logits(
            DLiteBatchPacket(
                input_ids,
                loss_mask,
                anchor_positions,
                block_keep_mask,
                target_hidden,
                history,
                (prediction_hidden),
            ),
            phase,
            metadata,
            components.lm_head,
        )
        supervised = get_training_adapter("dlite").training_step(
            fsdp_model,
            dict(
                input_ids=input_ids,
                loss_mask=loss_mask,
                target_hidden=target_hidden,
                raw_history_hidden=history,
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                target_prefill_logits=target_logits,
                target_logits_are_gathered=True,
                collect_detailed_metrics=collect_detailed_metrics,
            ),
        )
        loss = supervised.loss
        numerator = supervised.loss_terms.numerator
        denominator = supervised.loss_terms.denominator

        metric_output = supervised
    return metric_output


def parse_args(*, mode, argv=None):
    return {
        "sft": parse_sft_args,
        "teacher": parse_teacher_args,
        "two_stage": parse_two_stage_args,
    }[mode](argv)


class DLiteTrainingAdapter(TrainingAdapter):
    """Recipe for DLite training phases and model adapter.
    Replays skipped packets on training restart, waits for phase barriers
    before starting a new phase, and produces checkpoint metadata.
    """

    replay_skipped = False
    phase_barrier = True

    def required_target_layers(self, model):
        required_layer_ids = super().required_target_layers(model)
        if model.is_teacher:
            required_layer_ids.update(model.history_layer_ids)
        return required_layer_ids

    def build_config(self, args, *, source_config=None, model_role=None):
        return build_draft_config(
            args, source_config=source_config, model_role=model_role
        )

    def phases(self, args):
        if args.mode == "two_stage":
            return [
                Phase(
                    "stage1",
                    1,
                    args.stage1_epochs,
                    True,
                    False,
                    log_at_schedule_end=False,
                    checkpoint_prefix="stage1/",
                    final_checkpoint="stage1/final",
                ),
                Phase(
                    "transition",
                    2,
                    1,  # transition epochs, ddefault=1
                    True,
                    False,
                    log_at_schedule_end=False,
                    checkpoint_prefix="transition/",
                    final_checkpoint=(
                        "transition/final" if args.disaggregate else "transition"
                    ),
                    start_checkpoint=None if args.disaggregate else "transition/start",
                ),
                Phase(
                    "stage2",
                    3,
                    args.stage2_epochs,
                    False,
                    True,
                    log_at_schedule_end=not args.disaggregate,
                    checkpoint_prefix="stage2/",
                ),
            ]
        return [
            Phase(
                args.mode,
                1,
                args.num_epochs,
                args.mode == "teacher",
                args.mode == "sft",
                getattr(args, "max_steps", None),
                log_at_schedule_end=not args.disaggregate,
                schedule_full_epochs=not args.disaggregate,
                checkpoint_prefix=f"{args.mode}/" if args.disaggregate else "",
            )
        ]

    def initialize(self, session):
        args = session.args
        if args.mode == "teacher":
            from torch._inductor import config as inductor_config

            inductor_config.triton.autotune_pointwise = (
                os.environ.get("DLITE_POINTWISE_AUTOTUNE", "0") == "1"
            )
        draft, teacher, _ = _initialize_drafts(args, args.mode, session.resume_state)
        if args.mode == "teacher" and not draft.is_teacher:
            raise ValueError("Teacher training requires an swa_teacher checkpoint")
        if teacher is not None:
            if not teacher.is_teacher:
                raise ValueError("--teacher-draft-path must contain an swa_teacher")
            # SWA/history are teacher-only: compare shared dimensions, not roles.
            if (
                draft.block_size,
                draft.chs_num_layers,
                draft.sequential_head_type,
                draft.sequential_rank,
                draft.config.vocab_size,
                draft.config.hidden_size,
                tuple(draft.target_layer_ids),
            ) != (
                teacher.block_size,
                teacher.chs_num_layers,
                teacher.sequential_head_type,
                teacher.sequential_rank,
                teacher.config.vocab_size,
                teacher.config.hidden_size,
                tuple(teacher.target_layer_ids),
            ):
                raise ValueError(
                    "Teacher and student structures must match except draft depth"
                )
            teacher.requires_grad_(False)
        draft.requires_grad_(True)
        session.draft, session.teacher = draft, teacher
        provided = getattr(args, "teacher_draft_path", None)
        saved = (session.resume_state or {}).get("teacher_checkpoint_identity")
        session.teacher_identity = saved or (
            os.path.realpath(provided) if provided else None
        )
        if teacher is not None and saved and saved != os.path.realpath(provided):
            raise ValueError("Stage1/transition requires the same teacher checkpoint")
        validate_resume_model(session.resume_state, draft)

    def make_online(self, session):
        if session.teacher is not None:
            session.teacher_online = _make_online(
                session.args,
                session.teacher,
                session.components,
                session.process_group,
                teacher_defaults=True,
            )
        return _make_online(
            session.args, session.draft, session.components, session.process_group
        )

    def build_loader(self, session, tokenizer):
        args = session.args
        drafts = [session.draft] + (
            [session.teacher] if session.teacher is not None else []
        )
        return build_train_dataloader(
            args,
            tokenizer,
            train_data_path=args.train_data_path,
            cache_namespace={"sft": "sft", "teacher": "single", "two_stage": "train"}[
                args.mode
            ],
            num_proc=args.build_dataset_num_proc,
            required_layer_ids=required_hidden_layer_ids(drafts),
        )

    def optimizer_parameters(self, session):
        if session.args.mode == "two_stage":
            return two_stage_optimizer_parameters(session.draft)
        return None

    def enter_phase(self, session, phase):
        metrics = {}
        if session.args.mode == "two_stage":
            if phase.name == "stage1":
                _set_student_stage1_trainable(session.draft)
            else:
                _set_student_stage2_trainable(session.draft)
            if phase.name == "stage2":
                session.teacher_online = None
                session.teacher = None
                if session.target is not None:
                    session.target.set_capture_layers(session.draft.target_layer_ids)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if session.topology is None:
                        torch.cuda.reset_peak_memory_stats()
                        metrics = {
                            "detail/dlite/post_teacher_release_allocated_gib": torch.cuda.memory_allocated()
                            / 1024**3,
                            "detail/dlite/post_teacher_release_reserved_gib": torch.cuda.memory_reserved()
                            / 1024**3,
                        }
        session.draft.train()
        if session.teacher is not None:
            session.teacher.eval()
        return metrics

    def compute_step(
        self,
        session,
        data,
        *,
        microbatch_start=None,
        microbatch_end=None,
        collect_detailed_metrics=False,
    ):
        if session.topology is not None:
            return _packet_step(
                session,
                data,
                microbatch_start,
                microbatch_end,
                collect_detailed_metrics=collect_detailed_metrics,
            )
        if session.phase.name == "stage1":
            return _distillation_step(
                session, data, collect_detailed_metrics=collect_detailed_metrics
            )
        if session.phase.name == "transition":
            return _transition_step(
                session, data, collect_detailed_metrics=collect_detailed_metrics
            )
        return _supervised_step(
            session, data, collect_detailed_metrics=collect_detailed_metrics
        )

    def packet_metadata(self, session):
        return _draft_metadata(session.draft, session.teacher)

    def target_proxies(self, session):
        metadata = session.packet_metadata
        capture_layer_ids = set(metadata["target_layer_ids"]) | set(
            metadata["history_layer_ids"]
        )
        if any(phase.need_prediction_hidden for phase in session.phases):
            capture_layer_ids.add(metadata["num_target_layers"] - 1)
        return [
            SimpleNamespace(
                target_layer_ids=sorted(capture_layer_ids),
                is_teacher=bool(metadata["history_layer_ids"]),
                history_layer_ids=metadata["history_layer_ids"],
            )
        ]

    def packet_spec(self, session, phase):
        return _packet_spec(
            session.args, session.topology, phase, session.packet_metadata
        )

    def empty_packet(self, spec):
        return DLiteBatchPacket.empty(spec, device="cuda")

    def produce_packet(self, session, phase, values, batch_id):
        return _produce_packet(
            session.args,
            session.topology,
            session.target,
            phase,
            session.packet_metadata,
            values,
            batch_id,
        )

    def target_epoch_offset(self, phase):
        return phase.phase_id * 1_000_000

    def training_metadata(self, session):
        args = session.args
        metadata = {
            "loss_config": {
                name: value
                for name, value in vars(args).items()
                if name.startswith("stage1_")
                or name.startswith("stage2_")
                or name
                in (
                    "final_ce_weight",
                    "l1_loss_weight",
                    "base_lm_ce_weight",
                    "loss_decay_gamma",
                    "base_lm_ce_decay_gamma",
                )
            }
        }
        if args.mode == "two_stage":
            metadata.update(
                optimizer_parameter_order=TWO_STAGE_PARAMETER_ORDER,
                continuous_two_stage_scheduler=True,
                transition_epochs=1,
                serial_head_inherited=True,
                teacher_checkpoint_identity=session.teacher_identity,
            )
        return metadata

    def log_metrics(self, session, reduced):
        if session.topology is None:
            return reduced
        if session.phase.name == "stage1":
            return {key: reduced[key] for key in ("objective/loss", "kl")}
        if session.phase.name == "transition":
            return reduced
        return {
            "objective/loss": reduced["objective/loss"],
            "accuracy": reduced["hard_label/token_accuracy"],
            "hard_label/block_accepted_length": reduced[
                "hard_label/block_accepted_length"
            ],
            "expected_accept_len": reduced["expected_accept_len"],
            "final_ce": reduced["objective/final_ce"],
            "base_ce": reduced.get("objective/base_ce", 0.0),
            "l1_distance": reduced.get("objective/l1_distance", 0.0),
        }

    def checkpoint_metadata(self, model):
        metadata = super().checkpoint_metadata(model)
        metadata.update(
            architecture_version=model.architecture_version,
            model_role=model.model_role,
            swa_window_size=model.swa_window_size,
            chs_num_layers=model.chs_num_layers,
            sequential_head=model.sequential_head_type,
            sequential_rank=model.sequential_rank,
        )
        return metadata


ADAPTER = DLiteTrainingAdapter(
    "dlite", "DLiteDraftModel", ("sft", "teacher", "two_stage")
)
