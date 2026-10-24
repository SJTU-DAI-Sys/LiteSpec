#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/run_benchmark_dflash_family.sh" dflash2 "$@"
