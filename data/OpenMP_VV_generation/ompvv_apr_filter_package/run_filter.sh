#!/usr/bin/env bash
set -euo pipefail

INPUT=${1:-"/content/ompvv_bugs_failing_only_taxonomy (1).csv"}
OUTDIR=${2:-"/content/ompvv_final_apr_outputs"}
REVIEW_UPDATES=${3:-""}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "$REVIEW_UPDATES" ]]; then
  python "$SCRIPT_DIR/scripts/filter_ompvv_final_apr.py" \
    --input "$INPUT" \
    --outdir "$OUTDIR" \
    --review-updates "$REVIEW_UPDATES"
else
  python "$SCRIPT_DIR/scripts/filter_ompvv_final_apr.py" \
    --input "$INPUT" \
    --outdir "$OUTDIR"
fi
