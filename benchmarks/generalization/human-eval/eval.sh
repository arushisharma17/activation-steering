#!/usr/bin/env bash
#
# Portable evaluation runner for HumanEval generations
#
# Assumes:
#   - You have already generated JSONL files via gen.py into results/
#   - Your current Python environment has `human_eval` and `pandas`
#
# Optional:
#   export PROJECT_ROOT=/path/to/repo
#   export RESULTS_DIR=/path/to/results
#

set -euo pipefail

# Infer project root relative to this script, unless overridden
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

HE_DIR="${PROJECT_ROOT}/benchmarks/generalization/human-eval"
RESULTS_DIR="${RESULTS_DIR:-${HE_DIR}/results}"

if [[ ! -d "$HE_DIR" ]]; then
  echo "[ERROR] HumanEval directory not found: $HE_DIR"
  echo "Set PROJECT_ROOT to your repo root (export PROJECT_ROOT=/path/to/repo)."
  exit 1
fi

cd "$HE_DIR"
mkdir -p logs

echo "[INFO] Running evaluation in: $(pwd)"
echo "[INFO] Using results dir: $RESULTS_DIR"

python eval.py --results_dir "$RESULTS_DIR"

echo "[INFO] Done."

