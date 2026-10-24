#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -z "${PYTHON_BIN:-}" && -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

: "${TARGET_MODEL:?set TARGET_MODEL}"
: "${DRAFT_NAME_OR_PATH:?set DRAFT_NAME_OR_PATH}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29502}"
DATASET="${DATASET:-gsm8k}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"

OPTIONAL_ARGS=()
[[ -n "${VERIFY_BLOCK:-}" ]] && OPTIONAL_ARGS+=(--verify-block "${VERIFY_BLOCK}")
[[ "${COMPILE_SEQUENTIAL_HEAD:-0}" == "1" ]] && OPTIONAL_ARGS+=(--compile-serial-head)
[[ -n "${STOCHASTIC_VERIFICATION_MODE:-}" ]] && OPTIONAL_ARGS+=(
  --stochastic-verification-mode "${STOCHASTIC_VERIFICATION_MODE}"
)

CMD=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --nproc_per_node "${NPROC_PER_NODE}"
  --master_port "${MASTER_PORT}"
  evaluation/benchmark.py
  --model-name-or-path "${TARGET_MODEL}"
  --draft-name-or-path "${DRAFT_NAME_OR_PATH}"
  --dataset "${DATASET}"
  --max-samples "${MAX_SAMPLES}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --batch-size "${BATCH_SIZE}"
  --temperature "${TEMPERATURE}"
  "${OPTIONAL_ARGS[@]}"
)

printf 'DLite benchmark: target=%s draft=%s dataset=%s samples=%s\n' \
  "${TARGET_MODEL}" "${DRAFT_NAME_OR_PATH}" "${DATASET}" "${MAX_SAMPLES}"
printf 'Launching:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi
exec "${CMD[@]}"
