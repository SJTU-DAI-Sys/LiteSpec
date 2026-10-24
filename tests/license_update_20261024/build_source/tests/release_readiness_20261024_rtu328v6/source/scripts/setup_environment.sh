#!/usr/bin/env bash
# Install one complete, locked CUDA runtime. Shared environments are not modified.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cuda_variant="${1:-cu130}"
case "$cuda_variant" in
    cu128|cu130) ;;
    *) echo 'Usage: scripts/setup_environment.sh [cu128|cu130] [environment-path]' >&2; exit 2 ;;
esac
export UV_PROJECT_ENVIRONMENT="${2:-$repo_root/.venv-$cuda_variant}"
if [[ "$cuda_variant" == cu130 ]]; then
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
fi
export PATH="${CUDA_HOME:-/usr/local/cuda}/bin:$HOME/.local/bin:$PATH"
cd "$repo_root"
uv sync --frozen --extra "$cuda_variant" --group dev
uv pip check --python "$UV_PROJECT_ENVIRONMENT/bin/python"
