#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"
CONFIG="${PROJECT_DIR}/scripts/config/dlite_teacher.yaml"
if (( $# > 0 )) && [[ "$1" == *.yml || "$1" == *.yaml ]]; then
  CONFIG="$1"
  shift
fi
if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

cd "${PROJECT_DIR}"
exec "${PYTHON_BIN}" scripts/config/launch_dlite.py teacher "${CONFIG}" "$@"
