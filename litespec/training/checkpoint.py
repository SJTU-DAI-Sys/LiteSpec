"""Self-contained model exports and strict rank-local training checkpoints."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

from litespec.modeling.draft.registry import draft_source_files
from litespec.training.algorithms import get_training_adapter


def capture_rng_state():
    import random

    import numpy as np

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    import random

    import numpy as np

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def session_metadata(session):
    from dataclasses import asdict

    from .setup import topology_identity, training_data_identity

    state = session.state
    phase = session.phases[min(state.phase_index, len(session.phases) - 1)]
    metadata = {
        "cursor": asdict(state),
        "training_stage": phase.name,
        "phase_plan": [asdict(phase) for phase in session.phases],
        "steps_per_epoch": session.steps_per_epoch,
        "optimizer_step": state.optimizer_step,
        "scheduler_total_steps": session.total_steps,
        "scheduler_learning_rate": float(session.args.learning_rate),
        "scheduler_warmup_ratio": float(session.args.warmup_ratio),
        "train_data_identity": training_data_identity(session.args),
        "train_data_mode": (
            "regen_full" if session.args.train_hidden_states_path else "online"
        ),
        **session.adapter.training_metadata(session),
    }
    if getattr(session.args, "use_muon", False):
        from .muon import muon_config_from_args

        metadata["optimizer_kind"] = "muon_adamw_v1"
        metadata["muon_config"] = muon_config_from_args(session.args)
        metadata["scheduler_learning_rate"] = metadata["muon_config"]["adamw_lr"]
    if session.topology is not None:
        metadata.update(topology_identity(session.args, session.topology))
    return metadata


def validate_session_resume(session):
    if session.resume_state is None:
        return
    expected = session_metadata(session)
    if session.resume_state.get("optimizer_kind", "adamw") != expected.get(
        "optimizer_kind", "adamw"
    ):
        raise ValueError("Cannot resume across AdamW-only and Muon/AdamW modes")
    # These describe a particular update, not the immutable training contract.
    progress_keys = {
        "cursor",
        "training_stage",
        "optimizer_step",
        "selector_schedule_step",
        "selector_effective_alpha",
    }
    for key, value in expected.items():
        if key not in progress_keys and session.resume_state.get(key) != value:
            raise ValueError(
                f"Resume mismatch for {key}: saved={session.resume_state.get(key)!r}, "
                f"requested={value!r}"
            )
    from .algorithms import RunState

    cursor = RunState(**session.resume_state["cursor"])
    counts = [phase.batch_count(session.steps_per_epoch) for phase in session.phases]
    if not 0 <= cursor.phase_index <= len(counts):
        raise ValueError("Invalid checkpoint phase index")
    if (
        min(
            cursor.epoch,
            cursor.next_batch,
            cursor.phase_step,
            cursor.global_step,
            cursor.optimizer_step,
        )
        < 0
    ):
        raise ValueError("Negative checkpoint cursor")
    if cursor.global_step != sum(counts[: cursor.phase_index]) + cursor.phase_step:
        raise ValueError("Checkpoint global/phase cursors disagree")
    if cursor.phase_index < len(counts):
        if (
            cursor.phase_step > counts[cursor.phase_index]
            or cursor.phase_step
            != cursor.epoch * session.steps_per_epoch + cursor.next_batch
        ):
            raise ValueError("Checkpoint epoch/batch cursors disagree")
        if cursor.next_batch >= session.steps_per_epoch:
            raise ValueError("Checkpoint batch cursor is not normalized")
    elif cursor.phase_step or cursor.epoch or cursor.next_batch:
        raise ValueError("Completed checkpoint has an unfinished phase cursor")
    import math

    accumulation = session.args.accumulation_steps
    expected_updates = sum(
        math.ceil(count / accumulation) for count in counts[: cursor.phase_index]
    )
    if cursor.phase_step % accumulation:
        raise ValueError("Checkpoint is not at an optimizer boundary")
    expected_updates += cursor.phase_step // accumulation
    if (
        cursor.optimizer_step != expected_updates
        or session.resume_state["optimizer_step"] != expected_updates
    ):
        raise ValueError("Checkpoint optimizer step disagrees with the data cursor")
    session.state = cursor


def save_session(session, name):
    return save_checkpoint(
        output_dir=session.args.output_dir,
        name=name,
        fsdp_model=session.fsdp_model,
        draft_model=session.draft,
        optimizer=session.optimizer,
        metadata=session_metadata(session),
        training_args=session.args,
        training_yaml=session.training_yaml,
        process_group=session.process_group,
        coordinator_global_rank=session.coordinator_rank,
    )


_DISTRIBUTED_STATE_KEY = "distributed_training_state"


_DISTRIBUTED_STATE_FORMAT = "rank_local_optimizer_v2"


_OPTIMIZER_STATE_KEY = "optimizer_state_dict"


def ranked_training_state_path(checkpoint_dir: str, rank: int) -> str:
    """Return the training-state filename owned by one global rank."""
    return os.path.join(checkpoint_dir, f"training_state_rank_{rank:05d}.pt")


def distributed_training_state_exists(checkpoint_dir: str) -> bool:
    """Return whether the checkpoint's committed/common state exists."""
    return os.path.isfile(os.path.join(checkpoint_dir, "training_state.pt"))


def _rank_and_world_size(
    rank: int | None = None,
    world_size: int | None = None,
    process_group: dist.ProcessGroup | None = None,
) -> tuple[int, int]:
    if rank is None:
        rank = dist.get_rank(process_group) if dist.is_initialized() else 0
    if world_size is None:
        world_size = dist.get_world_size(process_group) if dist.is_initialized() else 1
    return rank, world_size


def _atomic_torch_save(state: dict[str, Any], path: str) -> None:
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def save_distributed_training_state(
    checkpoint_dir: str,
    state: dict[str, Any],
    *,
    rank: int | None = None,
    world_size: int | None = None,
    process_group: dist.ProcessGroup | None = None,
) -> None:
    """Save rank-local optimizer state and compact common metadata.

    In distributed runs, all ranks must call this function.  Each rank first
    writes its local optimizer shard. Only after every shard is durable does
    rank 0 publish a metadata-only ``training_state.pt``. Single-rank runs use
    the same layout so the common file never duplicates optimizer tensors.
    """
    rank, world_size = _rank_and_world_size(rank, world_size, process_group)
    os.makedirs(checkpoint_dir, exist_ok=True)

    state_to_save = dict(state)
    state_to_save[_DISTRIBUTED_STATE_KEY] = {
        "format": _DISTRIBUTED_STATE_FORMAT,
        "rank": rank,
        "world_size": world_size,
    }

    _atomic_torch_save(state_to_save, ranked_training_state_path(checkpoint_dir, rank))

    if dist.is_initialized():
        dist.barrier(group=process_group)

    if rank == 0:
        common_state = dict(state_to_save)
        common_state.pop(_OPTIMIZER_STATE_KEY, None)
        common_state.pop("fp32_master_params", None)
        common_state.pop("rng_state", None)
        common_state[_DISTRIBUTED_STATE_KEY] = {
            "format": _DISTRIBUTED_STATE_FORMAT,
            "rank": None,
            "world_size": world_size,
            "metadata_only": True,
        }
        _atomic_torch_save(
            common_state, os.path.join(checkpoint_dir, "training_state.pt")
        )

    if dist.is_initialized():
        dist.barrier(group=process_group)


def load_distributed_training_state(
    checkpoint_dir: str,
    *,
    map_location: Any = "cpu",
    rank: int | None = None,
    world_size: int | None = None,
    process_group: dist.ProcessGroup | None = None,
) -> dict[str, Any] | None:
    """Load a complete rank-local optimizer shard in the current format."""
    rank, world_size = _rank_and_world_size(rank, world_size, process_group)
    ranked_path = ranked_training_state_path(checkpoint_dir, rank)
    common_path = os.path.join(checkpoint_dir, "training_state.pt")
    if not os.path.isfile(common_path):
        return None
    if not os.path.isfile(ranked_path):
        raise FileNotFoundError(
            f"Optimizer state for global rank {rank} is missing. Expected {ranked_path}."
        )
    for other_rank in range(world_size):
        required_state_path = ranked_training_state_path(checkpoint_dir, other_rank)
        if not os.path.isfile(required_state_path):
            raise FileNotFoundError(
                f"Optimizer state for global rank {other_rank} is missing. "
                f"Expected {required_state_path}."
            )
    common = torch.load(common_path, map_location="cpu", weights_only=False)
    common_metadata = common.get(_DISTRIBUTED_STATE_KEY, {})
    if common_metadata.get("format") != _DISTRIBUTED_STATE_FORMAT:
        raise RuntimeError("Unsupported common distributed training-state format")
    if common_metadata.get("world_size") != world_size:
        raise RuntimeError("Cannot restore with a different world size")
    state_path = ranked_path
    state = torch.load(state_path, map_location=map_location, weights_only=False)
    metadata = state.get(_DISTRIBUTED_STATE_KEY)
    if metadata is None:
        raise ValueError(f"Missing distributed training-state metadata in {state_path}")

    if metadata.get("format") != _DISTRIBUTED_STATE_FORMAT:
        raise RuntimeError(
            f"Unsupported distributed training-state format in {state_path}: "
            f"{metadata.get('format')!r}."
        )
    saved_world_size = metadata.get("world_size")
    saved_rank = metadata.get("rank")
    if saved_world_size != world_size:
        raise RuntimeError(
            "Cannot restore rank-local optimizer state with a different world "
            f"size: checkpoint={saved_world_size}, current={world_size}."
        )
    if saved_rank != rank:
        expected_path = ranked_training_state_path(checkpoint_dir, rank)
        raise RuntimeError(
            f"Optimizer state for global rank {rank} is missing. Expected "
            f"{expected_path}, but found state for rank {saved_rank}."
        )
    for key in (
        "training_state_format",
        "cursor",
        "optimizer_step",
        "phase_plan",
        "optimizer_parameter_names",
    ):
        if state.get(key) != common.get(key):
            raise ValueError(f"Rank-local and common checkpoint disagree on {key}")
    return state


def export_draft(model, save_dir, *, state_dict=None):
    """Export a self-contained model without importing the training package."""
    model.config.auto_map = {
        "AutoModel": f"{type(model).__module__.rsplit('.', 1)[-1]}.{type(model).__name__}"
    }
    model.config.architectures = [type(model).__name__]
    source_root = Path(__file__).resolve().parents[1] / "modeling" / "draft"
    sources = draft_source_files(type(model).__name__)
    model.save_pretrained(save_dir, state_dict=state_dict)
    for filename in sources:
        shutil.copyfile(source_root / filename, Path(save_dir) / filename)
    # Transformers 4.57 copies only direct imports into the dynamic cache for
    # local directories. Declare the complete closure in the exported entry so
    # transitive helpers are copied too; repository sources stay unmodified.
    entry = type(model).__module__.rsplit(".", 1)[-1] + ".py"
    entry_path = Path(save_dir) / entry
    dependencies = "\n".join(
        f"from .{Path(filename).stem} import __name__ as _export_{Path(filename).stem}"
        for filename in sources
        if filename != entry
    )
    with entry_path.open("a") as stream:
        stream.write(
            "\n# Complete dependency list for the remote-code loader.\n"
            + dependencies
            + "\n"
        )


def save_checkpoint(
    *,
    output_dir: str,
    name: str,
    fsdp_model: FSDP,
    draft_model,
    optimizer,
    metadata: dict,
    training_args,
    training_yaml: str | None = None,
    process_group=None,
    coordinator_global_rank: int = 0,
) -> str:
    save_dir = os.path.join(output_dir, name)
    group_rank = dist.get_rank(process_group)
    if group_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier(group=process_group)
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT):
        full_state = fsdp_model.state_dict()
        draft_state = {
            key.split("draft_model.", 1)[1]: value
            for key, value in full_state.items()
            if "draft_model." in key
        }
        algorithm = draft_model.method_config_key.removesuffix("_config")
        model_metadata = get_training_adapter(algorithm).checkpoint_metadata(
            draft_model
        )
        model_metadata.update(training_identity(training_args))
        save_distributed_training_state(
            save_dir,
            {
                **metadata,
                **model_metadata,
                **optimizer.state_dict(),
                "rng_state": capture_rng_state(),
            },
            process_group=process_group,
        )
        if group_rank == 0:
            export_draft(draft_model, save_dir, state_dict=draft_state)
            if training_yaml is not None:
                Path(save_dir, "training_config.yaml").write_bytes(
                    training_yaml.encode("utf-8")
                )
    dist.barrier(group=process_group)
    if dist.get_rank() == coordinator_global_rank:
        print(f"Saved checkpoint to {save_dir}", flush=True)
    return save_dir


def load_training_state(checkpoint_dir: Optional[str], *, args=None) -> Optional[dict]:
    if checkpoint_dir is None:
        return None
    state = load_distributed_training_state(checkpoint_dir, map_location="cpu")
    if state is None:
        raise FileNotFoundError(
            f"No training_state.pt was found in checkpoint {checkpoint_dir!r}."
        )
    validate_training_identity(state, args)
    return state


def training_identity(args):
    return {
        "training_state_format": "litespec_training_v2",
        "algorithm": args.algorithm,
        "mode": args.mode if args.algorithm == "dlite" else None,
        "tp_size": int(args.tp_size),
        "shard_draft_by_tp": bool(args.shard_draft_by_tp),
        "execution_mode": "disaggregate" if args.disaggregate else "ordinary",
        "accumulation_steps": int(args.accumulation_steps),
        "batch_size": int(args.batch_size),
        "node_batch_size": args.node_batch_size if args.disaggregate else None,
        "draft_micro_batch_size": (
            args.draft_micro_batch_size if args.disaggregate else None
        ),
        "target_model_path": os.path.realpath(args.target_model_path),
    }


def validate_training_identity(state, args=None):
    if state.get("training_state_format") != "litespec_training_v2":
        raise ValueError("Resume requires the current LiteSpec training-state format")
    if args is not None:
        for key, value in training_identity(args).items():
            if state[key] != value:
                raise ValueError(
                    f"Resume mismatch for {key}: saved={state[key]!r}, requested={value!r}"
                )


def reserve_output_dir(output_dir: str) -> str:
    """Atomically reserve a non-existing output directory on rank zero."""

    output_dir = os.path.normpath(os.path.expanduser(output_dir))
    selected = None
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank == 0:
        suffix = 0
        while selected is None:
            candidate = output_dir if suffix == 0 else f"{output_dir}_{suffix}"
            try:
                os.makedirs(candidate, exist_ok=False)
            except FileExistsError:
                suffix += 1
            else:
                selected = candidate
    if dist.is_available() and dist.is_initialized():
        payload = [selected]
        dist.broadcast_object_list(payload, src=0)
        selected = payload[0]
        dist.barrier()
    assert selected is not None
    if rank == 0:
        print(f"Training output directory: {selected}", flush=True)
    return selected
