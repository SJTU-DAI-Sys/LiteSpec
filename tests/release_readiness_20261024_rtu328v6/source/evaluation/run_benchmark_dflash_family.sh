#!/usr/bin/env bash
set -euo pipefail
ALGORITHM="${1:?usage: run_benchmark_dflash_family.sh ALGORITHM}"
shift
case "${ALGORITHM}" in dflash|dflash2|dspark) ;; *) exit 2 ;; esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
: "${TARGET_MODEL:?set TARGET_MODEL}"
: "${DRAFT_NAME_OR_PATH:?set DRAFT_NAME_OR_PATH}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
CMD=("${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node "${NPROC_PER_NODE:-1}" --master_port "${MASTER_PORT:-29502}" evaluation/benchmark_dflash_family.py --algorithm "${ALGORITHM}" --model-name-or-path "${TARGET_MODEL}" --draft-name-or-path "${DRAFT_NAME_OR_PATH}" --dataset "${DATASET:-gsm8k}" --max-samples "${MAX_SAMPLES:-10}" --max-new-tokens "${MAX_NEW_TOKENS:-4096}" --temperature "${TEMPERATURE:-0.0}" "$@")
printf '%s benchmark:' "${ALGORITHM}"; printf ' %q' "${CMD[@]}"; printf '\n'
if [[ "${DRY_RUN:-0}" == "1" ]]; then exit 0; fi
exec "${CMD[@]}"
