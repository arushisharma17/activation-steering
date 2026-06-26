#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:-data/ompvv_bugs_june.json}"
OUTDIR="${2:-outputs}"

mkdir -p "$OUTDIR"

python scripts/00_filter_validated_bugs.py \
  --input "$INPUT" \
  --output-valid "$OUTDIR/ompvv_bugs_failing_only.json" \
  --output-rejected "$OUTDIR/ompvv_bugs_passing_rejected.json" \
  --summary "$OUTDIR/filter_summary.csv"

python scripts/01_extract_bug_types.py \
  --input "$OUTDIR/ompvv_bugs_failing_only.json" \
  --output "$OUTDIR/bug_type_template_failing_only.csv"

python scripts/02_build_mapping.py \
  --template "$OUTDIR/bug_type_template_failing_only.csv" \
  --output "$OUTDIR/bug_type_mapping_final_failing_only.csv"

python scripts/03_apply_taxonomy.py \
  --input "$OUTDIR/ompvv_bugs_failing_only.json" \
  --mapping "$OUTDIR/bug_type_mapping_final_failing_only.csv" \
  --output-json "$OUTDIR/ompvv_bugs_failing_only_taxonomy.json" \
  --output-csv "$OUTDIR/ompvv_bugs_failing_only_taxonomy.csv"

python scripts/04_compute_taxonomy_stats.py \
  --input "$OUTDIR/ompvv_bugs_failing_only_taxonomy.csv" \
  --outdir "$OUTDIR/stats_failing_only"

echo "Done. Final outputs written to $OUTDIR"
