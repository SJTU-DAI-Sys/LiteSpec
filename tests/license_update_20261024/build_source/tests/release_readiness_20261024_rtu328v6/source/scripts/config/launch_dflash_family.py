#!/usr/bin/env python3
"""Launch a DFlash-family training job from a YAML configuration."""

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


ALGORITHMS = {"dflash", "dflash2", "dspark"}


def load_config(path: Path, algorithm: str) -> tuple[dict[str, str], list[str]]:
    if algorithm not in ALGORITHMS:
        raise ConfigError(f"unsupported algorithm: {algorithm}")
    launcher, arguments, yaml_snapshot = load_yaml_config(
        path, identity_key="algorithm", identity_value=algorithm
    )
    env = launcher_environment(launcher, yaml_snapshot=yaml_snapshot)
    cli = arguments_to_cli(arguments)
    return env, cli


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        print("usage: launch_dflash_family.py ALGORITHM CONFIG.yaml", file=sys.stderr)
        return 2
    algorithm, config_value, *passthrough = argv
    project_dir = Path(__file__).resolve().parents[2]
    config_path = Path(config_value)
    if not config_path.is_absolute():
        config_path = project_dir / config_path
    try:
        configured_env, cli = load_config(config_path, algorithm)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return launch_training(
        label=algorithm,
        module=f"scripts.{algorithm}.train",
        configured_env=configured_env,
        configured_cli=cli,
        passthrough=passthrough,
        project_dir=project_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
