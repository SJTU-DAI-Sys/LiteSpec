"""Shared construction and training helpers for DFlash-family entrypoints."""

from __future__ import annotations

import argparse
import copy

import torch

from litespec.core.dflash import OnlineDFlashModel
from litespec.core.dflash2 import OnlineDFlash2Model
from litespec.core.dspark import OnlineDSparkModel
from litespec.distributed.transport import (
    DFlashFamilyBatchPacket,
    DFlashFamilyPacketSpec,
)
from litespec.modeling.config_utils import load_text_model_config
from litespec.modeling.draft.dflash import build_target_layer_ids
from litespec.training.algorithms import TrainingAdapter, get_training_adapter
from litespec.training.config import add_common_args, validate_common_args
from litespec.training.logging import get_tracker_class
from litespec.training.setup import (
    build_target_and_components,
    build_train_dataloader,
    hidden_states_to_cuda,
    required_hidden_layer_ids,
)

DFLASH_FAMILY_ALGORITHMS = ("dflash", "dflash2", "dspark")


class DFlashFamilyTrainingAdapter(TrainingAdapter):
    def build_config(self, args, *, source_config=None, model_role=None):
        return build_dflash_family_config(
            args, self.algorithm, source_config=source_config
        )

    def prepare_batch(self, data, target_output, draft, require_last):
        return prepare_dflash_family_features(data, target_output, draft, require_last)

    def phases(self, args):
        from litespec.training.algorithms import Phase

        return [Phase(self.algorithm, 1, int(args.num_epochs))]

    def initialize(self, session):
        args = session.args
        source = args.resume_from or args.init_from
        draft = (
            load_dflash_family_draft(source, self.algorithm, args.attention_backend)
            if source
            else build_dflash_family_draft(args, self.algorithm)
        )
        sync_args_from_draft(args, draft)
        draft.requires_grad_(True)
        session.draft = draft

    def make_online(self, session):
        return build_online_model(
            session.args,
            self.algorithm,
            session.draft,
            session.components,
            process_group=session.process_group,
        )

    def build_loader(self, session, tokenizer):
        return build_dflash_family_dataloader(
            session.args, tokenizer, session.draft, self.algorithm
        )

    def packet_metadata(self, session):
        draft = session.draft
        return {
            "algorithm": self.algorithm,
            "target_layer_ids": list(draft.target_layer_ids),
            "num_target_layers": int(draft.config.num_target_layers),
            "hidden_size": int(draft.config.hidden_size),
            "context_width": len(draft.target_layer_ids)
            * int(draft.config.hidden_size),
            "block_size": int(draft.block_size),
            "include_target_last_hidden": bool(session.args.require_target_last_hidden),
        }

    def target_proxies(self, session):
        from types import SimpleNamespace

        session.args.require_target_last_hidden = session.packet_metadata[
            "include_target_last_hidden"
        ]
        return [
            SimpleNamespace(
                target_layer_ids=session.packet_metadata["target_layer_ids"],
                is_teacher=False,
            )
        ]

    def packet_spec(self, session, phase):
        return _packet_spec(session.args, session.topology, session.packet_metadata)

    def empty_packet(self, spec):
        from litespec.distributed.transport import DFlashFamilyBatchPacket

        return DFlashFamilyBatchPacket.empty(spec, device="cuda")

    def produce_packet(self, session, phase, values, batch_id):
        return _target_packet(
            session.args,
            session.topology,
            session.target,
            session.packet_metadata,
            values,
        )

    def target_epoch_offset(self, phase):
        return 0

    replay_skipped = True
    phase_barrier = False

    def training_metadata(self, session):
        return dflash_family_training_metadata(
            session.args,
            self.algorithm,
            session.state.optimizer_step,
            session.total_steps,
        )

    def detail_metrics(self, session):
        if self.algorithm == "dflash2":
            # Called after optimizer.step, but the objective used the preceding step.
            return {
                "selector_alpha": selector_alpha(
                    session.args, session.state.optimizer_step - 1, session.total_steps
                )
            }
        return {}

    def compute_step(
        self,
        session,
        data,
        *,
        microbatch_start=None,
        microbatch_end=None,
        collect_detailed_metrics=False,
    ):
        args, draft = session.args, session.draft
        if session.topology is not None:
            input_ids = data.input_ids[microbatch_start:microbatch_end]
            loss_mask = data.loss_mask[microbatch_start:microbatch_end]
            target_context_hidden = data.target_context_hidden[
                microbatch_start:microbatch_end
            ]
            target_last_hidden = (
                data.target_last_hidden[microbatch_start:microbatch_end]
                if data.target_last_hidden is not None
                else None
            )
        else:
            from litespec.training.setup import select_tp_rank_batch

            input_ids = data["input_ids"].cuda(non_blocking=True)
            attention_mask = data["attention_mask"].cuda(non_blocking=True)
            loss_mask = data["loss_mask"].cuda(non_blocking=True)
            target_output = None
            if session.target is not None:
                target_output = session.target.generate_training_data(
                    input_ids, attention_mask, loss_mask, return_logits=False
                )
            target_context_hidden, target_last_hidden = self.prepare_batch(
                data, target_output, draft, args.require_target_last_hidden
            )
            if session.tp_draft_rank is not None:
                input_ids, loss_mask, target_context_hidden = (
                    select_tp_rank_batch(value, session.tp_draft_rank)
                    for value in (input_ids, loss_mask, target_context_hidden)
                )
                if target_last_hidden is not None:
                    target_last_hidden = select_tp_rank_batch(
                        target_last_hidden, session.tp_draft_rank
                    )
        kwargs = dict(
            input_ids=input_ids,
            hidden_states=target_context_hidden,
            loss_mask=loss_mask,
            target_last_hidden_states=target_last_hidden,
            collect_detailed_metrics=collect_detailed_metrics,
        )
        if self.algorithm == "dflash2":
            kwargs["selector_loss_alpha"] = selector_alpha(
                args, session.state.optimizer_step, session.total_steps
            )
        return self.training_step(session.fsdp_model, kwargs)


def add_dflash_family_args(parser: argparse.ArgumentParser, algorithm: str) -> None:
    if algorithm not in DFLASH_FAMILY_ALGORITHMS:
        raise ValueError(f"Unsupported DFlash-family algorithm: {algorithm}")
    add_common_args(parser)
    defaults = {"dflash": 16, "dflash2": 16, "dspark": 7}
    parser.set_defaults(block_size=defaults[algorithm], num_draft_layers=5)
    model = parser.add_argument_group(f"{algorithm} architecture")
    model.add_argument("--attention-mode", choices=("gqa", "mha"), default="gqa")
    model.add_argument("--conv-kernel-size", type=int, default=2)
    model.add_argument("--conv-group-size", type=int, default=16)
    model.add_argument("--selector-rank", type=int, default=256)
    model.add_argument("--selector-top-k", type=int, default=16)
    model.add_argument(
        "--markov-head-type", choices=("vanilla", "gated", "rnn"), default="vanilla"
    )
    model.add_argument("--markov-rank", type=int, default=256)
    model.add_argument(
        "--confidence-head", action=argparse.BooleanOptionalAction, default=True
    )
    model.add_argument(
        "--confidence-head-with-markov",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    model.add_argument(
        "--backbone-conv", action=argparse.BooleanOptionalAction, default=False
    )

    train = parser.add_argument_group(f"{algorithm} objective")
    train.add_argument("--init-from")
    train.add_argument("--num-epochs", type=int, default=6)
    train.add_argument("--learning-rate", type=float, default=6e-4)
    train.add_argument("--warmup-ratio", type=float, default=0.04)
    train.add_argument("--loss-decay-gamma", type=float)
    train.add_argument("--objective-chunk-blocks", type=int, default=128)
    train.add_argument("--selector-loss-alpha", type=float, default=1.0)
    train.add_argument("--selector-warmup-ratio", type=float, default=0.0)
    train.add_argument("--selector-ramp-ratio", type=float, default=0.0)
    train.add_argument(
        "--selector-stop-gradient", action=argparse.BooleanOptionalAction, default=False
    )
    train.add_argument("--ce-loss-alpha", type=float, default=0.1)
    train.add_argument("--l1-loss-alpha", type=float, default=0.9)
    train.add_argument("--confidence-loss-alpha", type=float, default=1.0)


def validate_dflash_family_args(
    parser: argparse.ArgumentParser, args, algorithm: str
) -> None:
    validate_common_args(parser, args)
    if args.resume_from and args.init_from:
        parser.error("--resume-from and --init-from are mutually exclusive")
    if args.num_epochs <= 0 or args.learning_rate <= 0:
        parser.error("--num-epochs and --learning-rate must be positive")
    if not 0 <= args.warmup_ratio <= 1:
        parser.error("--warmup-ratio must be in [0, 1]")
    if args.objective_chunk_blocks < 0:
        parser.error("--objective-chunk-blocks must be non-negative")
    if algorithm == "dflash2":
        if args.conv_kernel_size < 1 or args.conv_kernel_size > args.block_size:
            parser.error("--conv-kernel-size must be in [1, block-size]")
        if args.selector_rank < 1 or args.selector_top_k < 1:
            parser.error("selector rank and top-k must be positive")
        if args.selector_loss_alpha < 0:
            parser.error("--selector-loss-alpha must be non-negative")
        if not 0 <= args.selector_warmup_ratio <= 1:
            parser.error("--selector-warmup-ratio must be in [0, 1]")
        if not 0 <= args.selector_ramp_ratio <= 1:
            parser.error("--selector-ramp-ratio must be in [0, 1]")
    if algorithm == "dspark":
        if min(args.ce_loss_alpha, args.l1_loss_alpha, args.confidence_loss_alpha) < 0:
            parser.error("DSpark loss weights must be non-negative")
        if args.ce_loss_alpha + args.l1_loss_alpha + args.confidence_loss_alpha == 0:
            parser.error("At least one DSpark loss weight must be positive")
        if args.markov_rank < 1:
            parser.error("--markov-rank must be positive")
        if args.backbone_conv:
            if args.conv_kernel_size < 1 or args.conv_kernel_size > args.block_size:
                parser.error("--conv-kernel-size must be in [1, block-size]")
            if args.conv_group_size < 1:
                parser.error("--conv-group-size must be positive")
    # Every family records expected_accept_len from the target distribution and
    # the algorithm's final proposal distribution.
    args.require_target_last_hidden = True


def _parse_target_layer_ids(args, config) -> list[int]:
    if args.target_layer_ids:
        ids = [
            int(value.strip())
            for value in args.target_layer_ids.split(",")
            if value.strip()
        ]
        if ids != sorted(set(ids)):
            raise ValueError("--target-layer-ids must be unique and increasing")
        if not ids or ids[0] < 0 or ids[-1] >= int(config.num_target_layers):
            raise ValueError("--target-layer-ids contains an out-of-range layer")
        return ids
    return build_target_layer_ids(
        int(config.num_target_layers), int(config.num_hidden_layers)
    )


def build_dflash_family_config(args, algorithm: str, source_config=None):
    config = (
        load_text_model_config(
            args.target_model_path, trust_remote_code=args.trust_remote_code
        )
        if source_config is None
        else copy.deepcopy(source_config)
    )
    if source_config is None:
        target_depth = int(config.num_hidden_layers)
        config.num_hidden_layers = int(args.num_draft_layers)
        config.num_target_layers = target_depth
    config.block_size = int(args.block_size)
    config.architectures = [get_training_adapter(algorithm).model_class.__name__]
    config.auto_map = {
        "AutoModel": f"{algorithm}.{get_training_adapter(algorithm).model_class.__name__}"
    }
    config._attn_implementation = args.attention_backend
    config.layer_types = ["full_attention"] * int(config.num_hidden_layers)
    config.sliding_window = None
    config.use_sliding_window = False
    if args.attention_mode == "mha":
        config.num_key_value_heads = int(config.num_attention_heads)
    method = dict(getattr(config, f"{algorithm}_config", None) or {})
    method.update(
        {
            "block_size": int(args.block_size),
            "target_layer_ids": _parse_target_layer_ids(args, config),
            "mask_token_id": args.mask_token_id,
            "attention_mode": args.attention_mode,
        }
    )
    if algorithm == "dflash2":
        method.update(
            {
                "conv_kernel_size": int(args.conv_kernel_size),
                "conv_group_size": int(args.conv_group_size),
                "selector_rank": int(args.selector_rank),
                "selector_top_k": int(args.selector_top_k),
            }
        )
    elif algorithm == "dspark":
        method.update(
            {
                "projector_type": "dspark",
                "markov_head_type": args.markov_head_type,
                "markov_rank": int(args.markov_rank),
                "enable_confidence_head": bool(args.confidence_head),
                "confidence_head_with_markov": bool(args.confidence_head_with_markov),
                "confidence_head_alpha": float(args.confidence_loss_alpha),
                "dspark_backbone_conv_enabled": bool(args.backbone_conv),
            }
        )
        if args.backbone_conv:
            method.update(
                {
                    "conv_kernel_size": int(args.conv_kernel_size),
                    "conv_group_size": int(args.conv_group_size),
                }
            )
    setattr(config, f"{algorithm}_config", method)
    return config


def build_dflash_family_draft(args, algorithm: str, source_config=None):
    return get_training_adapter(algorithm).build_model(
        args, source_config=source_config
    )


def load_dflash_family_draft(path: str, algorithm: str, attention_backend: str):
    return get_training_adapter(algorithm).load_model(path, attention_backend)


def sync_args_from_draft(args, draft) -> None:
    method = dict(getattr(draft.config, draft.method_config_key) or {})
    args.block_size = int(draft.block_size)
    args.num_draft_layers = int(draft.config.num_hidden_layers)
    args.target_layer_ids = ",".join(str(value) for value in draft.target_layer_ids)
    args.attention_mode = str(method.get("attention_mode", "gqa"))
    draft.config._attn_implementation = args.attention_backend


def concatenate_target_hidden(
    hidden_states, target_layer_ids: list[int]
) -> torch.Tensor:
    if isinstance(hidden_states, dict):
        selected = [hidden_states[int(layer_id)] for layer_id in target_layer_ids]
    else:
        offset = 1 if len(hidden_states) > max(target_layer_ids) + 1 else 0
        selected = [
            hidden_states[int(layer_id) + offset] for layer_id in target_layer_ids
        ]
    return torch.cat(selected, dim=-1)


def final_target_hidden(hidden_states, num_target_layers: int) -> torch.Tensor:
    layer_id = int(num_target_layers) - 1
    if isinstance(hidden_states, dict):
        return hidden_states[layer_id]
    offset = 1 if len(hidden_states) == int(num_target_layers) + 1 else 0
    return hidden_states[layer_id + offset]


def build_dflash_family_components(args, draft):
    return build_target_and_components(args, [draft])


def build_dflash_family_dataloader(args, tokenizer, draft, algorithm: str):
    required_layer_ids = required_hidden_layer_ids([draft])
    if args.require_target_last_hidden:
        required_layer_ids.add(int(draft.config.num_target_layers) - 1)
    return build_train_dataloader(
        args,
        tokenizer,
        train_data_path=args.train_data_path,
        cache_namespace=algorithm,
        num_proc=args.build_dataset_num_proc,
        required_layer_ids=required_layer_ids,
    )


def build_online_model(args, algorithm: str, draft, components, process_group=None):
    common = dict(
        draft_model=draft,
        target_lm_head=components.lm_head,
        target_embed_tokens=components.embed_tokens,
        mask_token_id=int(args.mask_token_id),
        block_size=int(draft.block_size),
        attention_backend=args.attention_backend,
        flex_attention_backend=args.flex_attention_backend,
        num_anchors=int(args.num_anchors),
        loss_decay_gamma=args.loss_decay_gamma,
        objective_chunk_blocks=int(args.objective_chunk_blocks),
        process_group=process_group,
    )
    if algorithm == "dspark":
        return OnlineDSparkModel(
            **common,
            dspark_ce_loss_alpha=float(args.ce_loss_alpha),
            dspark_l1_loss_alpha=float(args.l1_loss_alpha),
            dspark_confidence_head_alpha=float(args.confidence_loss_alpha),
        )
    online_class = OnlineDFlash2Model if algorithm == "dflash2" else OnlineDFlashModel
    return online_class(
        **common,
        selector_loss_alpha=(
            float(args.selector_loss_alpha) if algorithm == "dflash2" else 0.0
        ),
        selector_warmup_ratio=float(args.selector_warmup_ratio),
        selector_ramp_ratio=float(args.selector_ramp_ratio),
        selector_stop_gradient=bool(args.selector_stop_gradient),
    )


def prepare_dflash_family_features(data, target_output, draft, require_last: bool):
    target_hidden_states = (
        hidden_states_to_cuda(data["hidden_states"])
        if target_output is None
        else hidden_states_to_cuda(target_output.hidden_states)
    )
    target_context_hidden = concatenate_target_hidden(
        target_hidden_states, list(draft.target_layer_ids)
    )
    target_last_hidden = (
        final_target_hidden(target_hidden_states, int(draft.config.num_target_layers))
        if require_last
        else None
    )
    return target_context_hidden, target_last_hidden


def selector_alpha(args, optimizer_step: int, total_steps: int) -> float:
    target_selector_alpha = float(args.selector_loss_alpha)
    warmup_steps = int(total_steps * float(args.selector_warmup_ratio))
    ramp_steps = int(total_steps * float(args.selector_ramp_ratio))
    if optimizer_step < warmup_steps:
        return 0.0
    if ramp_steps > 0 and optimizer_step < warmup_steps + ramp_steps:
        return target_selector_alpha * (optimizer_step - warmup_steps + 1) / ramp_steps
    return target_selector_alpha


def dflash_family_training_metadata(
    args, algorithm: str, optimizer_step: int, total_steps: int
) -> dict:
    """Serializable objective and schedule state stored in every checkpoint."""

    objective = {
        "loss_decay_gamma": args.loss_decay_gamma,
        "objective_chunk_blocks": int(args.objective_chunk_blocks),
    }
    if algorithm == "dflash2":
        objective.update(
            {
                "selector_loss_alpha": float(args.selector_loss_alpha),
                "selector_warmup_ratio": float(args.selector_warmup_ratio),
                "selector_ramp_ratio": float(args.selector_ramp_ratio),
                "selector_stop_gradient": bool(args.selector_stop_gradient),
            }
        )
    elif algorithm == "dspark":
        objective.update(
            {
                "ce_loss_alpha": float(args.ce_loss_alpha),
                "l1_loss_alpha": float(args.l1_loss_alpha),
                "confidence_loss_alpha": float(args.confidence_loss_alpha),
            }
        )
    return {
        "loss_config": objective,
        "selector_schedule_step": int(optimizer_step),
        "selector_effective_alpha": (
            selector_alpha(args, optimizer_step, total_steps)
            if algorithm == "dflash2"
            else 0.0
        ),
    }


__all__ = [
    "DFLASH_FAMILY_ALGORITHMS",
    "add_dflash_family_args",
    "build_dflash_family_components",
    "build_dflash_family_config",
    "build_dflash_family_dataloader",
    "build_dflash_family_draft",
    "build_online_model",
    "concatenate_target_hidden",
    "final_target_hidden",
    "dflash_family_training_metadata",
    "load_dflash_family_draft",
    "prepare_dflash_family_features",
    "selector_alpha",
    "sync_args_from_draft",
    "validate_dflash_family_args",
]


def parse_dflash_family_args(algorithm: str, argv=None):
    parser = argparse.ArgumentParser(description=f"Train {algorithm} draft model")
    add_dflash_family_args(parser, algorithm)
    args = parser.parse_args(argv)
    args.algorithm = algorithm
    validate_dflash_family_args(parser, args, algorithm)
    get_tracker_class(args.report_to).validate_args(parser, args)
    return args


def _packet_spec(args, topology, metadata):
    local_batch_size = (
        int(args.node_batch_size) // topology.target_replicas_per_node
        if topology.is_target
        else int(args.node_batch_size) // topology.draft_ranks_per_node
    )
    return DFlashFamilyPacketSpec(
        batch_size=local_batch_size,
        max_length=int(args.max_length),
        context_width=int(metadata["context_width"]),
        hidden_size=int(metadata["hidden_size"]),
        include_target_last_hidden=bool(metadata["include_target_last_hidden"]),
    )


def _target_packet(args, topology, target, metadata, values):
    input_ids, attention_mask, loss_mask = values
    target_output = target.generate_training_data(
        input_ids, attention_mask, loss_mask, return_logits=False
    )
    if not topology.is_target_leader:
        return None
    target_hidden_states = target_output.hidden_states
    target_context_hidden = concatenate_target_hidden(
        target_hidden_states, metadata["target_layer_ids"]
    )
    target_last_hidden = (
        final_target_hidden(target_hidden_states, metadata["num_target_layers"])
        if metadata["include_target_last_hidden"]
        else None
    )
    return DFlashFamilyBatchPacket(
        input_ids=input_ids,
        loss_mask=loss_mask,
        target_context_hidden=target_context_hidden.to(torch.bfloat16),
        target_last_hidden=(
            target_last_hidden.to(torch.bfloat16)
            if target_last_hidden is not None
            else None
        ),
    )
