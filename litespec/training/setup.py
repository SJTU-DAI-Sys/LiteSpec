"""Data, target, topology and FSDP preparation; no training loops."""

from __future__ import annotations

import hashlib
import math
import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    MixedPrecision,
    ShardingStrategy,
)
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from datasets import load_dataset
from litespec.args import SGLangBackendArgs
from litespec.data import (
    RegenFullCollator,
    RegenFullDataset,
    ShardAwareDistributedSampler,
    build_training_dataset,
    load_hidden_cache_manifest,
    prepare_dp_dataloaders,
)
from litespec.distributed import get_dp_group, get_tp_group, print_on_rank0
from litespec.modeling.config_utils import is_qwen35_model_type, load_text_model_config
from litespec.modeling.target.target_utils import (
    SGLangTPEmbeddingAdapter,
    SGLangTPLMHeadAdapter,
    SharedTargetEmbeddingsAndHead,
    TargetEmbeddingsAndHead,
)
from litespec.modeling.target.training_target_model import get_training_target_model
from litespec.training.checkpoint import validate_training_identity


def _build_processed_dataset(
    args,
    tokenizer,
    *,
    train_data_path: str,
    cache_namespace: str,
    num_proc: Optional[int] = None,
):
    cache_key = hashlib.md5(
        (
            f"{train_data_path}-{args.max_length}-{args.chat_template}-"
            f"{args.target_model_path}-preformatted={args.is_preformatted}"
        ).encode()
    ).hexdigest()
    raw_dataset = load_dataset("json", data_files=train_data_path)["train"]
    return build_training_dataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        is_preformatted=args.is_preformatted,
        cache_dir=os.path.join(args.cache_dir, cache_namespace, "processed_dataset"),
        cache_key=cache_key,
        num_proc=(args.build_dataset_num_proc if num_proc is None else int(num_proc)),
    )


def _has_valid_anchor_supervision(value, *, block_size: int) -> bool:
    """Match the anchor sampler's per-example supervision requirements."""
    loss_mask = torch.as_tensor(value["loss_mask"]).reshape(-1)
    minimum = 2 * int(block_size)
    if int(loss_mask.sum().item()) < minimum:
        return False
    max_anchor = int(loss_mask.numel()) - int(block_size)
    if max_anchor < 0:
        return False
    current = loss_mask[: max_anchor + 1] > 0.5
    following = loss_mask[1 : max_anchor + 2] > 0.5
    return bool((current & following).any().item())


def _prepare_dataloader(args, dataset, *, train_data_path: str):
    minimum = 2 * int(args.block_size)
    dataset = dataset.filter(
        _has_valid_anchor_supervision,
        fn_kwargs={"block_size": int(args.block_size)},
        desc=(
            "Filtering examples with at least "
            f"{minimum} labels and one trainable anchor"
        ),
    )
    dataloader = prepare_dp_dataloaders(
        dataset,
        args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        process_group=get_dp_group(),
        pad_to_length=args.max_length if args.pad_to_max_length else None,
    )
    if len(dataloader) == 0:
        raise ValueError(
            f"Training dataset {train_data_path!r} has no full batches after filtering."
        )
    return dataloader


def build_train_dataloader(
    args,
    tokenizer,
    *,
    train_data_path: Optional[str] = None,
    cache_namespace: str = "single",
    num_proc: Optional[int] = None,
    required_layer_ids: Optional[set[int]] = None,
):
    if args.train_hidden_states_path:
        if train_data_path is not None:
            raise ValueError("online train_data_path cannot be used in offline mode")
        return build_hidden_cache_dataloader(
            args,
            tokenizer,
            required_layer_ids=set(required_layer_ids or ()),
        )
    train_data_path = train_data_path or args.train_data_path
    if not train_data_path:
        raise ValueError("A training data path is required.")
    dataset = None
    if dist.get_rank() == 0:
        dataset = _build_processed_dataset(
            args,
            tokenizer,
            train_data_path=train_data_path,
            cache_namespace=cache_namespace,
            num_proc=num_proc,
        )
    dist.barrier()
    if dataset is None:
        dataset = _build_processed_dataset(
            args,
            tokenizer,
            train_data_path=train_data_path,
            cache_namespace=cache_namespace,
            num_proc=num_proc,
        )
    return _prepare_dataloader(args, dataset, train_data_path=train_data_path)


def build_hidden_cache_dataloader(args, tokenizer, *, required_layer_ids: set[int]):
    target_config = load_text_model_config(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    verify = not dist.is_initialized() or dist.get_rank() == 0
    manifest = load_hidden_cache_manifest(
        args.train_hidden_states_path,
        target_model=args.target_model_path,
        num_hidden_layers=int(target_config.num_hidden_layers),
        hidden_size=int(target_config.hidden_size),
        required_layer_ids=required_layer_ids,
        max_length=int(args.max_length),
        verify_checksums=verify,
    )
    if dist.is_initialized():
        dist.barrier()
        if not verify:
            manifest = load_hidden_cache_manifest(
                args.train_hidden_states_path,
                target_model=args.target_model_path,
                num_hidden_layers=int(target_config.num_hidden_layers),
                hidden_size=int(target_config.hidden_size),
                required_layer_ids=required_layer_ids,
                max_length=int(args.max_length),
                verify_checksums=False,
            )
    dataset = RegenFullDataset(
        args.train_hidden_states_path,
        manifest,
        required_layer_ids=required_layer_ids,
        max_length=args.max_length,
    )
    # This reads only manifest JSON metadata. Sample tensor members remain unopened
    # until a worker fetches a batch.
    valid_indices = dataset.eligible_indices(block_size=args.block_size)
    process_group = get_dp_group()
    sampler = ShardAwareDistributedSampler(
        dataset,
        valid_indices,
        num_replicas=dist.get_world_size(process_group),
        rank=dist.get_rank(process_group),
        shuffle=True,
        seed=args.seed,
    )
    if args.dataloader_num_workers == 0:
        prefetch_factor = None
    else:
        prefetch_factor = 2
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.dataloader_num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        persistent_workers=args.dataloader_num_workers > 0,
        collate_fn=RegenFullCollator(
            tokenizer.pad_token_id or 0,
            args.max_length if args.pad_to_max_length else None,
        ),
        drop_last=True,
    )
    if len(dataloader) == 0:
        raise ValueError(
            f"Training cache {args.train_hidden_states_path!r} has no full batches "
            "after filtering."
        )
    return dataloader


def training_data_identity(args) -> str:
    path = args.train_hidden_states_path or args.train_data_path
    identity = os.path.realpath(path)
    if args.train_hidden_states_path:
        manifest_path = os.path.join(identity, "manifest.json")
        digest = hashlib.sha256()
        with open(manifest_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        identity = f"{identity}#manifest_sha256={digest.hexdigest()}"
    return identity


def required_hidden_layer_ids(draft_models) -> set[int]:
    from .algorithms import get_training_adapter

    required_layer_ids: set[int] = set()
    for draft in draft_models:
        adapter = get_training_adapter(draft.method_config_key.removesuffix("_config"))
        required_layer_ids.update(adapter.required_target_layers(draft))
    return required_layer_ids


def hidden_states_to_cuda(hidden_states):
    if isinstance(hidden_states, dict):
        return {
            key: value.cuda(non_blocking=True) for key, value in hidden_states.items()
        }
    return tuple(value.cuda(non_blocking=True) for value in hidden_states)


def project_cached_target_logits(
    final_hidden_state: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_size: int,
    lm_head: torch.nn.Module,
) -> torch.Tensor:
    """Project only cached positions needed by the DLite target loss."""
    offsets = torch.arange(int(block_size) - 1, device=anchor_positions.device).view(
        1, 1, -1
    )
    positions = anchor_positions.unsqueeze(-1) + offsets
    if bool((positions >= final_hidden_state.size(1)).any()):
        raise ValueError("Cached final hidden states do not cover target positions.")
    expanded = final_hidden_state.unsqueeze(1).expand(
        -1, anchor_positions.size(1), -1, -1
    )
    selected = torch.gather(
        expanded,
        2,
        positions.unsqueeze(-1).expand(-1, -1, -1, final_hidden_state.size(-1)),
    )
    with torch.no_grad():
        return lm_head(selected)


def load_cached_target_data(
    data,
    *,
    anchor_positions: torch.Tensor,
    block_size: int,
    lm_head: torch.nn.Module,
    need_logits: bool,
):
    hidden_states = hidden_states_to_cuda(data["hidden_states"])
    target_logits = None
    if need_logits:
        final_layer_id = int(data["final_norm_layer_id"])
        final_hidden = hidden_states[final_layer_id]
        target_logits = project_cached_target_logits(
            final_hidden, anchor_positions, block_size, lm_head
        )
    return hidden_states, target_logits


def select_tp_rank_batch(value, tp_rank: int):
    """Copy rank ``tp_rank``'s sample so full target tensors can be released."""
    if isinstance(value, dict):
        return {key: select_tp_rank_batch(item, tp_rank) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(select_tp_rank_batch(item, tp_rank) for item in value)
    if isinstance(value, list):
        return [select_tp_rank_batch(item, tp_rank) for item in value]
    if not isinstance(value, torch.Tensor) or value.ndim == 0:
        raise TypeError(
            "TP draft batch values must be batch-first tensors or containers."
        )
    if not 0 <= int(tp_rank) < value.size(0):
        raise ValueError(
            f"TP rank {tp_rank} cannot select from batch size {value.size(0)}."
        )
    return value.narrow(0, int(tp_rank), 1).clone(memory_format=torch.contiguous_format)


def stage_total_steps(dataloader, epochs: int, accumulation_steps: int) -> int:
    # The loops intentionally carry a partial accumulation across epoch
    # boundaries and flush only once at the end of the stage.  Therefore the
    # scheduler must count all micro-batches together rather than rounding each
    # epoch independently.
    return math.ceil(int(epochs) * len(dataloader) / int(accumulation_steps))


def build_target_model(args, draft_models):
    target_config = load_text_model_config(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    source_model_type = getattr(
        target_config, "target_source_model_type", target_config.model_type
    )
    if (
        is_qwen35_model_type(source_model_type)
        and args.target_model_backend != "sglang"
    ):
        raise ValueError(
            "Qwen3.5 targets require --target-model-backend sglang with the "
            "pinned Transformers version; the draft remains a dense "
            "Qwen3-style model."
        )
    backend_kwargs = {}
    if args.target_model_backend == "sglang":
        if is_qwen35_model_type(source_model_type):
            # SGLang 0.5.9's Qwen3.5 hybrid GDN path is validated with FA3;
            # FlashInfer attempts to JIT an incompatible fallback prefill path.
            args.sglang_attention_backend = "fa3"
        backend_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()
        if backend_kwargs["max_running_requests"] is None:
            backend_kwargs["max_running_requests"] = int(args.batch_size)
        if backend_kwargs["max_total_tokens"] is None:
            backend_kwargs["max_total_tokens"] = int(args.batch_size) * int(
                args.max_length
            )
    target = get_training_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=torch.bfloat16,
        device="cuda" if args.target_model_backend == "hf" else None,
        trust_remote_code=args.trust_remote_code,
        **backend_kwargs,
    )
    capture_layer_ids = required_hidden_layer_ids(draft_models)
    if getattr(args, "require_target_last_hidden", False):
        capture_layer_ids.add(int(target_config.num_hidden_layers) - 1)
    target.set_capture_layers(sorted(capture_layer_ids))
    return target


def build_target_and_components(args, draft_models):
    """Build one target and bind its tokenizer/embedding/head consistently."""
    offline = bool(args.train_hidden_states_path)
    target = None if offline else build_target_model(args, draft_models)
    tokenizer, components, mask_token_id = resolve_tokenizer_and_components(
        args,
        draft_models,
        target=target,
        standalone_components=offline,
    )
    return target, tokenizer, components, mask_token_id


def _select_mask_token_id(
    *,
    explicit_id: Optional[int],
    configured_ids: list[Optional[int]],
    tokenizer_mask_id: Optional[int],
    used_token_ids: set[int],
    vocab_size: int,
) -> tuple[int, str]:
    """Select a real embedding row without assuming a model-specific token ID."""
    vocab_size = int(vocab_size)
    if vocab_size <= 0:
        raise ValueError(f"Target vocabulary size must be positive, got {vocab_size}.")

    if explicit_id is not None:
        candidate = int(explicit_id)
        source = "explicit --mask-token-id"
    else:
        configured = {int(value) for value in configured_ids if value is not None}
        if len(configured) > 1:
            raise ValueError(
                "Draft checkpoints disagree on mask_token_id: "
                f"{sorted(configured)}. Pass --mask-token-id explicitly."
            )
        if configured:
            candidate = configured.pop()
            source = "draft checkpoint"
        elif tokenizer_mask_id is not None:
            candidate = int(tokenizer_mask_id)
            source = "tokenizer.mask_token_id"
        else:
            candidate = next(
                (
                    token_id
                    for token_id in range(vocab_size)
                    if token_id not in used_token_ids
                ),
                -1,
            )
            source = "first tokenizer-unused model vocabulary row"

    if not 0 <= candidate < vocab_size:
        if candidate < 0 and explicit_id is None:
            raise ValueError(
                "No tokenizer-unused row exists inside the target embedding vocabulary. "
                "Configure a model-provided mask token or pass --mask-token-id."
            )
        raise ValueError(
            f"MASK token id {candidate} selected from {source} is outside target "
            f"vocabulary [0, {vocab_size})."
        )
    return candidate, source


def resolve_tokenizer_and_components(
    args, draft_models, target=None, *, standalone_components: bool = False
):
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    target_config = load_text_model_config(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    configured_mask_ids = []
    for draft in draft_models:
        method_config = getattr(draft.config, draft.method_config_key)
        if method_config:
            configured_mask_ids.append(method_config.get("mask_token_id"))
    mask_token_id, mask_source = _select_mask_token_id(
        explicit_id=args.mask_token_id,
        configured_ids=configured_mask_ids,
        tokenizer_mask_id=tokenizer.mask_token_id,
        used_token_ids={int(value) for value in tokenizer.get_vocab().values()},
        vocab_size=int(target_config.vocab_size),
    )
    args.mask_token_id = mask_token_id
    print_on_rank0(
        f"Resolved MASK token id {mask_token_id} from {mask_source}; "
        f"target vocab_size={int(target_config.vocab_size)}."
    )
    if args.target_model_backend == "sglang" and not standalone_components:
        if target is None or not hasattr(target, "model_runner"):
            raise ValueError("SGLang target is required to reuse target components.")
        target_model = target.model_runner.model
        target_embedding = target_model.get_input_embeddings()
        target_lm_head = target_model.lm_head
        components = SharedTargetEmbeddingsAndHead(
            SGLangTPEmbeddingAdapter(target_embedding, get_tp_group(), mask_token_id),
            SGLangTPLMHeadAdapter(target_lm_head, get_tp_group()),
        )
        components.requires_grad_(False)
        print_on_rank0(
            "Reusing SGLang target-resident TP embedding and LM head; "
            "no independent full-vocabulary copies were loaded."
        )
    else:
        components = TargetEmbeddingsAndHead.from_pretrained(
            args.target_model_path,
            embed_key=args.embedding_key,
            lm_head_key=args.lm_head_key,
            device="cuda",
            trust_remote_code=args.trust_remote_code,
        )
        if not 0 <= mask_token_id < components.embed_tokens.num_embeddings:
            raise ValueError(
                "DLite vocab_row MASK mode requires an existing target "
                f"embedding row, but mask_token_id={mask_token_id} and target "
                f"vocab size={components.embed_tokens.num_embeddings}. Pass "
                "--mask-token-id with an unused in-vocabulary row."
            )
    for draft in draft_models:
        draft.mask_token_id = mask_token_id
        draft.mask_embedding_mode = "vocab_row"
        getattr(draft.config, draft.method_config_key)["mask_token_id"] = mask_token_id
    return tokenizer, components, mask_token_id


def wrap_training_model(draft_online, components, *, process_group=None):
    return FullyShardedDataParallel(
        draft_online,
        process_group=process_group,
        ignored_modules=[components.lm_head, components.embed_tokens],
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )


def validate_tp_draft_sharding(args) -> Optional[int]:
    """Validate one-rank/one-sample draft sharding and return the TP rank."""
    if not args.shard_draft_by_tp:
        return None
    if args.target_model_backend != "sglang":
        raise ValueError(
            "--shard-draft-by-tp requires --target-model-backend sglang so "
            "the target is actually tensor parallel."
        )
    tp_group = get_tp_group()
    tp_size = dist.get_world_size(tp_group)
    if tp_size <= 1:
        raise ValueError("--shard-draft-by-tp requires --tp-size > 1.")
    if int(args.batch_size) != tp_size:
        raise ValueError(
            "One-sample-per-TP-rank draft sharding requires target batch size "
            f"to equal tp_size; got batch_size={args.batch_size}, tp_size={tp_size}."
        )
    print_on_rank0(
        f"shard-draft-by-tp enabled: target batch={args.batch_size}; "
        "each TP rank trains one distinct draft sample."
    )
    return dist.get_rank(tp_group)


class FixedCollator:
    def __init__(self, max_length: int, pad_token_id: int):
        self.max_length = int(max_length)
        self.pad_token_id = int(pad_token_id)

    def _pad(self, value, *, fill=0, dtype=None):
        value = torch.as_tensor(value).reshape(-1)[: self.max_length]
        output = torch.full((self.max_length,), fill, dtype=dtype or value.dtype)
        output[: value.numel()].copy_(value.to(output.dtype))
        return output

    def __call__(self, features):
        return {
            "input_ids": torch.stack(
                [
                    self._pad(row["input_ids"], fill=self.pad_token_id)
                    for row in features
                ]
            ),
            "attention_mask": torch.stack(
                [self._pad(row["attention_mask"], fill=0) for row in features]
            ),
            "loss_mask": torch.stack(
                [
                    self._pad(row["loss_mask"], fill=0, dtype=torch.float32)
                    for row in features
                ]
            ),
        }


def topology_identity(args, topology) -> dict:
    return {
        "execution_mode": "disaggregate",
        "target_ranks_per_node": int(args.target_ranks_per_node),
        "draft_ranks_per_node": int(args.draft_ranks_per_node),
        "target_tp_size": int(args.target_tp_size),
        "target_ep_size": int(args.sglang_ep_size),
        "node_batch_size": int(args.node_batch_size),
        "global_batch_size": int(args.node_batch_size) * int(topology.nnodes),
        "draft_world_size": len(topology.draft_global_ranks),
    }


def validate_resume_topology(state: Optional[dict], identity: dict) -> None:
    if state is None:
        return
    for key, current in identity.items():
        saved = state[key]
        if saved != current:
            raise ValueError(
                f"Disaggregated resume topology mismatch for {key}: "
                f"checkpoint={saved!r}, current={current!r}"
            )


def _load_common_state(path: Optional[str], *, args=None) -> Optional[dict]:
    if not path:
        return None
    common = os.path.join(path, "training_state.pt")
    if not os.path.isfile(common):
        raise FileNotFoundError(f"No training_state.pt in {path!r}")
    state = torch.load(common, map_location="cpu", weights_only=False)
    validate_training_identity(state, args)
    return state


def _build_target_loader(args, tokenizer, topology):
    dataset = None
    if topology.rank == 0:
        dataset = _build_processed_dataset(
            args,
            tokenizer,
            train_data_path=args.train_data_path,
            cache_namespace="disaggregate",
            num_proc=args.build_dataset_num_proc,
        )
        dataset = dataset.filter(
            _has_valid_anchor_supervision,
            fn_kwargs={"block_size": int(args.block_size)},
        )
    dist.barrier()
    if topology.is_target_leader and topology.rank != 0:
        dataset = _build_processed_dataset(
            args,
            tokenizer,
            train_data_path=args.train_data_path,
            cache_namespace="disaggregate",
            num_proc=args.build_dataset_num_proc,
        )
        dataset = dataset.filter(
            _has_valid_anchor_supervision,
            fn_kwargs={"block_size": int(args.block_size)},
        )
    dataset_size = torch.tensor(
        [len(dataset) if topology.rank == 0 else 0], device="cuda", dtype=torch.long
    )
    dist.broadcast(dataset_size, src=0)
    producers_total = topology.nnodes * topology.target_replicas_per_node
    local_batch_size = int(args.node_batch_size) // topology.target_replicas_per_node
    steps_per_epoch = int(dataset_size.item()) // (producers_total * local_batch_size)
    if steps_per_epoch <= 0:
        raise ValueError("dataset is too small for one disaggregated global batch")
    dataloader = None
    if topology.is_target_leader:
        producer_rank = topology.node_rank * topology.target_replicas_per_node + int(
            topology.target_replica_local_rank
        )
        sampler = DistributedSampler(
            dataset,
            num_replicas=producers_total,
            rank=producer_rank,
            shuffle=True,
            drop_last=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=local_batch_size,
            sampler=sampler,
            num_workers=args.dataloader_num_workers,
            collate_fn=FixedCollator(
                args.max_length, tokenizer.pad_token_id or tokenizer.eos_token_id or 0
            ),
            drop_last=True,
            prefetch_factor=2 if args.dataloader_num_workers else None,
        )
    return dataloader, steps_per_epoch


def _broadcast_input(args, topology, data):
    shape = (int(args.target_batch_size), int(args.max_length))
    if topology.is_target_leader:
        values = (
            data["input_ids"].cuda(non_blocking=True),
            data["attention_mask"].cuda(non_blocking=True),
            data["loss_mask"].cuda(non_blocking=True),
        )
    else:
        values = (
            torch.empty(shape, dtype=torch.long, device="cuda"),
            torch.empty(shape, dtype=torch.long, device="cuda"),
            torch.empty(shape, dtype=torch.float32, device="cuda"),
        )
    for value in values:
        dist.broadcast(
            value,
            src=topology.target_tp_leader_global_rank,
            group=topology.target_tp_group,
        )
    return values


def build_training_session(args, adapter):
    """Prepare resources once; no training iteration or optimizer update here."""
    from litespec.distributed import init_disaggregated, init_distributed
    from litespec.distributed.transport import build_node_routes

    from .algorithms import TrainingSession
    from .checkpoint import (
        load_distributed_training_state,
        load_training_state,
        reserve_output_dir,
        validate_session_resume,
    )
    from .fsdp_layout import build_fsdp_layouts
    from .logging import create_tracker
    from .muon import muon_config_from_args
    from .optimizer import (
        BF16Optimizer,
        optimizer_step_count,
        register_optimizer_parameters,
    )

    session = TrainingSession(args, adapter, adapter.phases(args))
    if args.disaggregate:
        session.topology = init_disaggregated(
            timeout=args.dist_timeout,
            target_ranks_per_node=args.target_ranks_per_node,
            draft_ranks_per_node=args.draft_ranks_per_node,
            target_tp_size=args.target_tp_size,
            target_ep_size=args.sglang_ep_size,
        )
    else:
        init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
        session.tp_draft_rank = validate_tp_draft_sharding(args)
    args.output_dir = reserve_output_dir(args.output_dir)
    if args.disaggregate:
        topology = session.topology
        if topology.rank == 0:
            os.makedirs(args.cache_dir, exist_ok=True)
        dist.barrier()
        session.resume_state = _load_common_state(args.resume_from, args=args)
        validate_resume_topology(session.resume_state, topology_identity(args, topology))
        if topology.is_draft:
            adapter.initialize(session)
        source = topology.draft_global_ranks[0]
        payload = [
            adapter.packet_metadata(session) if topology.rank == source else None
        ]
        dist.broadcast_object_list(payload, src=source)
        session.packet_metadata = payload[0]
        args.block_size = session.packet_metadata["block_size"]
        tokenizer = AutoTokenizer.from_pretrained(
            args.target_model_path, trust_remote_code=args.trust_remote_code
        )
        session.dataloader, session.steps_per_epoch = _build_target_loader(
            args, tokenizer, topology
        )
        session.routes = build_node_routes(
            producers=topology.target_replicas_per_node,
            drafts=topology.draft_ranks_per_node,
            node_batch_size=args.node_batch_size,
        )
        if topology.is_target:
            # Validate the producer's schedule/cursor before starting any sends.
            from dataclasses import asdict

            if session.resume_state:
                for key, value in {
                    "phase_plan": [asdict(p) for p in session.phases],
                    "steps_per_epoch": session.steps_per_epoch,
                    "train_data_identity": training_data_identity(args),
                }.items():
                    if session.resume_state[key] != value:
                        raise ValueError(f"Resume mismatch for {key}")
            session.target = build_target_model(args, adapter.target_proxies(session))
            return session
        drafts = [session.draft] + (
            [session.teacher] if session.teacher is not None else []
        )
        _, session.components, _ = resolve_tokenizer_and_components(
            args, drafts, standalone_components=True
        )
        args.tracker_global_rank = session.coordinator_rank
    else:
        # resume from checkpoint
        session.resume_state = load_training_state(args.resume_from, args=args)
        adapter.initialize(session)
        drafts = [session.draft] + (
            [session.teacher] if session.teacher is not None else []
        )
        # target model and components
        session.target, tokenizer, session.components, _ = build_target_and_components(
            args, drafts
        )
        session.dataloader = adapter.build_loader(session, tokenizer)
        session.steps_per_epoch = len(session.dataloader)
    session.total_steps = optimizer_step_count(
        session.phases, session.steps_per_epoch, args.accumulation_steps
    )
    if session.total_steps <= 0:
        raise ValueError("The training schedule has no optimizer steps")
    session.draft_online = adapter.make_online(session)
    muon_config = muon_config_from_args(args)
    parameters = specs = layouts = None
    if muon_config is not None:
        parameters, specs = register_optimizer_parameters(
            session.draft, adapter.optimizer_parameters(session)
        )
    session.fsdp_model = wrap_training_model(
        session.draft_online, session.components, process_group=session.process_group
    )
    if muon_config is not None:
        layouts = build_fsdp_layouts(
            session.fsdp_model, parameters, specs, session.process_group
        )
    session.optimizer = BF16Optimizer(
        session.draft,
        parameters=(
            parameters
            if muon_config is not None
            else adapter.optimizer_parameters(session)
        ),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=session.total_steps,
        process_group=session.process_group,
        muon_config=muon_config,
        parameter_specs=specs,
        parameter_layouts=layouts,
    )
    validate_session_resume(session)
    if session.resume_state is not None:
        if args.disaggregate:
            session.resume_state = load_distributed_training_state(
                args.resume_from, process_group=session.process_group
            )
        session.optimizer.load_state_dict(session.resume_state)
    session.tracker = create_tracker(args, args.output_dir)
    # DataLoader iterator construction must not consume the model/anchor RNG.
    if session.dataloader is not None:
        session.dataloader.generator = torch.Generator().manual_seed(args.seed)
    return session
