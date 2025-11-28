#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=8G
#SBATCH --job-name="mcq_summary"
#SBATCH --mail-user=arushi17@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="logs/slurm-mcq-summary-%j.out"

########################################
# Environment Setup (Nova)
########################################

cd /lustre/hdd/LAS/jannesar-lab/arushi
source myenv/bin/activate

cd activation-steering/

########################################
# Optional CLI args
########################################
# Usage examples:
#   sbatch slurm/run_mcq_summary.sh
#   sbatch slurm/run_mcq_summary.sh \
#     --metrics mcq_cache/metrics_ab_apr.jsonl \
#     --flat mcq_cache/my_flat.csv \
#     --pairs mcq_cache/my_pairs.csv
########################################

METRICS_PATH="mcq_cache/metrics_ab_apr.jsonl"
OUT_FLAT="mcq_cache/metrics_flat.csv"
OUT_PAIRS="mcq_cache/metrics_pairs.csv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --metrics)
      METRICS_PATH="$2"
      shift 2
      ;;
    --flat)
      OUT_FLAT="$2"
      shift 2
      ;;
    --pairs)
      OUT_PAIRS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "============================================"
echo "Summarizing MCQ metrics"
echo "  Metrics JSONL: ${METRICS_PATH}"
echo "  Flat CSV     : ${OUT_FLAT}"
echo "  Paired CSV   : ${OUT_PAIRS}"
echo "============================================"

python scripts/summarize_ab_apr_metrics.py \
  --metrics_path "${METRICS_PATH}" \
  --output_flat "${OUT_FLAT}" \
  --output_pairs "${OUT_PAIRS}"

echo "[INFO] Summary done."

