"""Shared YAML and torchrun helpers for training launchers."""

from __future__ import annotations

import os
import shlex
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

COMMON_LAUNCHER_KEYS = {
    "target_model",
    "train_data",
    "train_hidden_states",
    "output_dir",
    "nproc_per_node",
    "nnodes",
    "node_rank",
    "master_addr",
    "master_port",
}


class ConfigError(ValueError):
    pass


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{location} must be a mapping")
    return value


def _positive_int(value: Any, location: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{location} must be a positive integer") from exc
    if parsed <= 0:
        raise ConfigError(f"{location} must be a positive integer")
    return parsed


def _nonnegative_int(value: Any, location: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{location} must be a non-negative integer") from exc
    if parsed < 0:
        raise ConfigError(f"{location} must be a non-negative integer")
    return parsed


def load_yaml_config(
    path: Path,
    *,
    identity_key: str,
    identity_value: str,
    extra_launcher_keys: Sequence[str] = (),
) -> tuple[Mapping[str, Any], Mapping[str, Any], str]:
    """Parse and snapshot the same YAML read, preserving comments and newlines."""
    if not path.is_file():
        raise ConfigError(f"config does not exist: {path}")
    try:
        source = path.read_bytes().decode("utf-8")
        config = yaml.safe_load(source)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"failed to read {path}: {exc}") from exc
    config = _mapping(config, "config")
    unknown = sorted(set(config) - {identity_key, "launcher", "arguments"})
    if unknown:
        raise ConfigError(f"unknown config keys: {unknown}")
    configured_identity = config.get(identity_key)
    if configured_identity != identity_value:
        raise ConfigError(
            f"config {identity_key} {configured_identity!r} does not match "
            f"launcher {identity_value!r}"
        )

    launcher = _mapping(config.get("launcher"), "launcher")
    allowed = COMMON_LAUNCHER_KEYS | set(extra_launcher_keys)
    unknown = sorted(set(launcher) - allowed)
    if unknown:
        raise ConfigError(f"unknown launcher keys: {unknown}")
    missing_field_names = [
        key for key in ("target_model", "output_dir") if not launcher.get(key)
    ]
    if missing_field_names:
        raise ConfigError(f"missing launcher keys: {missing_field_names}")
    if bool(launcher.get("train_data")) == bool(launcher.get("train_hidden_states")):
        raise ConfigError(
            "set exactly one of launcher.train_data and launcher.train_hidden_states"
        )

    arguments = _mapping(config.get("arguments", {}), "arguments")
    return launcher, arguments, source


def validate_common_settings(
    launcher: Mapping[str, Any], arguments: Mapping[str, Any]
) -> None:
    nproc_per_node = _positive_int(
        launcher.get("nproc_per_node", 8), "launcher.nproc_per_node"
    )
    nnodes = _positive_int(launcher.get("nnodes", 1), "launcher.nnodes")
    node_rank = _nonnegative_int(launcher.get("node_rank", 0), "launcher.node_rank")
    _positive_int(launcher.get("master_port", 29500), "launcher.master_port")
    if node_rank >= nnodes:
        raise ConfigError("launcher.node_rank must be in [0, launcher.nnodes)")

    tp_size = _positive_int(arguments.get("tp_size", 1), "arguments.tp_size")
    batch_size = _positive_int(arguments.get("batch_size", 1), "arguments.batch_size")
    shard = bool(arguments.get("shard_draft_by_tp", False))
    disaggregate = bool(arguments.get("disaggregate", False))
    offline = bool(launcher.get("train_hidden_states"))

    if shard:
        if offline:
            raise ConfigError("shard_draft_by_tp supports train_data only")
        if disaggregate:
            raise ConfigError("shard_draft_by_tp is incompatible with disaggregate")
        if arguments.get("target_model_backend", "hf") != "sglang":
            raise ConfigError("shard_draft_by_tp requires target_model_backend: sglang")
        if tp_size <= 1:
            raise ConfigError("shard_draft_by_tp requires tp_size > 1")
        if batch_size != tp_size:
            raise ConfigError("shard_draft_by_tp requires batch_size == tp_size")

    if not disaggregate:
        return
    if offline:
        raise ConfigError("disaggregate supports train_data only")
    required_arg_names = (
        "target_ranks_per_node",
        "draft_ranks_per_node",
        "node_batch_size",
    )
    missing_arg_names = [
        name for name in required_arg_names if arguments.get(name) is None
    ]
    if missing_arg_names:
        raise ConfigError(
            "disaggregate requires "
            + ", ".join(f"arguments.{name}" for name in missing_arg_names)
        )
    target_ranks = _positive_int(
        arguments["target_ranks_per_node"], "arguments.target_ranks_per_node"
    )
    draft_ranks = _positive_int(
        arguments["draft_ranks_per_node"], "arguments.draft_ranks_per_node"
    )
    if target_ranks + draft_ranks != nproc_per_node:
        raise ConfigError(
            "disaggregate requires target_ranks_per_node + draft_ranks_per_node "
            "== launcher.nproc_per_node"
        )
    target_tp = _positive_int(
        arguments.get("target_tp_size", tp_size), "arguments.target_tp_size"
    )
    ep_size = _positive_int(
        arguments.get("sglang_ep_size", 1), "arguments.sglang_ep_size"
    )
    node_batch = _positive_int(
        arguments["node_batch_size"], "arguments.node_batch_size"
    )
    pipeline_depth = _positive_int(
        arguments.get("pipeline_depth", 2), "arguments.pipeline_depth"
    )
    if pipeline_depth < 2:
        raise ConfigError("arguments.pipeline_depth must be at least 2")
    if target_ranks % target_tp:
        raise ConfigError("target_ranks_per_node must be divisible by target_tp_size")
    if target_tp % ep_size:
        raise ConfigError("target_tp_size must be divisible by sglang_ep_size")
    producers = target_ranks // target_tp
    if node_batch % producers:
        raise ConfigError("node_batch_size must be divisible by target producers")
    if node_batch % draft_ranks:
        raise ConfigError("node_batch_size must be divisible by draft_ranks_per_node")
    draft_batch = node_batch // draft_ranks
    draft_micro = _positive_int(
        arguments.get("draft_micro_batch_size", draft_batch),
        "arguments.draft_micro_batch_size",
    )
    if draft_batch % draft_micro:
        raise ConfigError("draft_micro_batch_size must divide the per-rank draft batch")
    if (target_tp > 1 or ep_size > 1) and arguments.get(
        "target_model_backend", "hf"
    ) != "sglang":
        raise ConfigError(
            "disaggregated TP/EP > 1 requires target_model_backend: sglang"
        )


def launcher_environment(
    launcher: Mapping[str, Any], *, yaml_snapshot: str
) -> dict[str, str]:
    env = {
        # torchrun workers inherit the launch-time contents, never a mutable path.
        "LITESPEC_TRAINING_YAML": yaml_snapshot,
        "TARGET_MODEL": str(launcher["target_model"]),
        "OUTPUT_DIR": str(launcher["output_dir"]),
        "NPROC_PER_NODE": str(launcher.get("nproc_per_node", 8)),
        "NNODES": str(launcher.get("nnodes", 1)),
        "NODE_RANK": str(launcher.get("node_rank", 0)),
        "MASTER_ADDR": str(launcher.get("master_addr", "127.0.0.1")),
        "MASTER_PORT": str(launcher.get("master_port", 29500)),
    }
    if launcher.get("train_data"):
        env["TRAIN_DATA_PATH"] = str(launcher["train_data"])
    else:
        env["TRAIN_HIDDEN_STATES_PATH"] = str(launcher["train_hidden_states"])
    return env


def arguments_to_cli(arguments: Mapping[str, Any]) -> list[str]:
    cli: list[str] = []
    for raw_name, value in arguments.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ConfigError("argument names must be non-empty strings")
        option = "--" + raw_name.strip().replace("_", "-")
        if value is None:
            continue
        if value is False:
            cli.append("--no-" + raw_name.strip().replace("_", "-"))
            continue
        cli.append(option)
        if value is True:
            continue
        if isinstance(value, list):
            if not value:
                raise ConfigError(f"{raw_name} list must not be empty")
            cli.extend(str(item) for item in value)
        elif isinstance(value, (str, int, float)):
            cli.append(str(value))
        else:
            raise ConfigError(
                f"argument {raw_name!r} has unsupported value type "
                f"{type(value).__name__}"
            )
    return cli


def training_base_cli(env: Mapping[str, str]) -> list[str]:
    cli = [
        "--target-model-path",
        env["TARGET_MODEL"],
        "--output-dir",
        env["OUTPUT_DIR"],
    ]
    if env.get("TRAIN_DATA_PATH"):
        cli.extend(["--train-data-path", env["TRAIN_DATA_PATH"]])
    else:
        cli.extend(["--train-hidden-states-path", env["TRAIN_HIDDEN_STATES_PATH"]])
    return cli


def effective_training_yaml(
    source: str,
    parsed_args: Mapping[str, Any],
    argument_names: Sequence[str],
    env: Mapping[str, str],
) -> str:
    """Merge typed trainer options and actual torchrun settings into the snapshot."""
    config = yaml.safe_load(source)
    launcher = config["launcher"]
    arguments = config.setdefault("arguments", {})
    launcher_options = {
        "target_model_path": "target_model",
        "train_data_path": "train_data",
        "train_hidden_states_path": "train_hidden_states",
        "output_dir": "output_dir",
        "teacher_draft_path": "teacher_draft",
    }
    for name in dict.fromkeys([*arguments, *argument_names]):
        value = parsed_args[name]
        if name in launcher_options:
            launcher[launcher_options[name]] = value
            arguments.pop(name, None)
        else:
            arguments[name] = value
    # Paths can also have been supplied through launcher fields, rather than
    # arguments. Record the parsed values to avoid stale duplicate settings.
    for name, key in launcher_options.items():
        if name in parsed_args and (parsed_args[name] is not None or key in launcher):
            launcher[key] = parsed_args[name]
    for key in ("nnodes", "node_rank", "nproc_per_node", "master_addr", "master_port"):
        value = env.get(f"PET_{key.upper()}", env[key.upper()])
        launcher[key] = value if key == "master_addr" else int(value)
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


def launch_training(
    *,
    label: str,
    module: str,
    configured_env: Mapping[str, str],
    configured_cli: Sequence[str],
    passthrough: Sequence[str],
    project_dir: Path,
) -> int:
    env = os.environ.copy()
    env.update(configured_env)
    venv_bin = str(Path(sys.executable).resolve().parent)
    env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONPATH"] = (
        f"{project_dir}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else str(project_dir)
    )
    trainer_cli = [*training_base_cli(env), *configured_cli, *passthrough]
    effective = {}
    for index, token in enumerate(trainer_cli):
        if not token.startswith("--"):
            continue
        key, separator, value = token[2:].partition("=")
        if key.startswith("no-"):
            effective[key[3:].replace("-", "_")] = False
        else:
            if not separator:
                value = (
                    trainer_cli[index + 1]
                    if index + 1 < len(trainer_cli)
                    and not trainer_cli[index + 1].startswith("--")
                    else True
                )
            effective[key.replace("-", "_")] = value
    topology = {
        "nproc_per_node": env.get("PET_NPROC_PER_NODE", env["NPROC_PER_NODE"]),
        "nnodes": env.get("PET_NNODES", env["NNODES"]),
        "node_rank": env.get("PET_NODE_RANK", env["NODE_RANK"]),
        "master_port": env.get("PET_MASTER_PORT", env["MASTER_PORT"]),
        "train_hidden_states": effective.get("train_hidden_states_path"),
    }
    validate_common_settings(topology, effective)
    if module == "scripts.dlite.train_two_stage":
        if not (effective.get("teacher_draft_path") or effective.get("resume_from")):
            raise ConfigError(
                "two_stage requires --teacher-draft-path or --resume-from"
            )
        for key in ("stage1_epochs", "stage2_epochs", "learning_rate"):
            if key not in effective:
                raise ConfigError(f"two_stage requires --{key.replace('_', '-')}")
    if env.get("DRY_RUN") != "1":
        sys.path.insert(0, str(project_dir))
        from litespec.training.algorithms import get_training_adapter

        is_dlite = module.startswith("scripts.dlite.")
        adapter = get_training_adapter("dlite" if is_dlite else label)
        parsed = adapter.parse_args(label if is_dlite else None, argv=trainer_cli)
        env["LITESPEC_TRAINING_YAML"] = effective_training_yaml(
            configured_env["LITESPEC_TRAINING_YAML"], vars(parsed), list(effective), env
        )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        env.get("PET_NNODES", env["NNODES"]),
        "--node_rank",
        env.get("PET_NODE_RANK", env["NODE_RANK"]),
        "--nproc_per_node",
        env.get("PET_NPROC_PER_NODE", env["NPROC_PER_NODE"]),
        "--master_addr",
        env.get("PET_MASTER_ADDR", env["MASTER_ADDR"]),
        "--master_port",
        env.get("PET_MASTER_PORT", env["MASTER_PORT"]),
        "-m",
        module,
        *trainer_cli,
    ]
    print(f"Launching {label}: {shlex.join(command)}", flush=True)
    if env.get("DRY_RUN") == "1":
        return 0
    os.chdir(project_dir)
    os.execvpe(sys.executable, command, env)
    return 0
