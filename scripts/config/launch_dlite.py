#!/usr/bin/env python3
"""Launch a DLite training mode from a YAML configuration."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from scripts.config.training_launcher import (
        ConfigError,
        arguments_to_cli,
        launch_training,
        launcher_environment,
        load_yaml_config,
    )
except ModuleNotFoundError:  # Direct execution adds scripts/config to sys.path.
    from training_launcher import (  # type: ignore[no-redef]
        ConfigError,
        arguments_to_cli,
        launch_training,
        launcher_environment,
        load_yaml_config,
    )


MODES = {"sft", "teacher", "two_stage"}
SCRIPTS = {
    "sft": "scripts.dlite.train_sft",
    "teacher": "scripts.dlite.train_teacher",
    "two_stage": "scripts.dlite.train_two_stage",
}


def load_config(path: Path, mode: str) -> tuple[dict[str, str], list[str]]:
    if mode not in MODES:
        raise ConfigError(f"unsupported DLite mode: {mode}")
    launcher, arguments, yaml_snapshot = load_yaml_config(
        path,
        identity_key="mode",
        identity_value=mode,
        extra_launcher_keys=("teacher_draft",),
    )
    env = launcher_environment(launcher, yaml_snapshot=yaml_snapshot)
    if launcher.get("teacher_draft"):
        env["TEACHER_DRAFT_PATH"] = str(launcher["teacher_draft"])
    cli = arguments_to_cli(arguments)
    return env, cli


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        print("usage: launch_dlite.py MODE CONFIG.yaml", file=sys.stderr)
        return 2
    mode, config_value, *passthrough = argv
    project_dir = Path(__file__).resolve().parents[2]
    config_path = Path(config_value)
    if not config_path.is_absolute():
        config_path = project_dir / config_path
    try:
        configured_env, cli = load_config(config_path, mode)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    extra_cli = []
    if configured_env.get("TEACHER_DRAFT_PATH"):
        extra_cli.extend(["--teacher-draft-path", configured_env["TEACHER_DRAFT_PATH"]])
    return launch_training(
        label=mode,
        module=SCRIPTS[mode],
        configured_env=configured_env,
        configured_cli=[*extra_cli, *cli],
        passthrough=passthrough,
        project_dir=project_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
