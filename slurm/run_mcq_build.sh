#!/bin/bash

#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:a100
#SBATCH --mem=64G
#SBATCH --job-name="mcq_build"
#SBATCH --mail-user=arushi17@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="logs/slurm-mcq-build-%j.out"

########################################
# Environment Setup (Nova)
########################################

export MPLCONFIGDIR=/lustre/hdd/LAS/jannesar-lab/arushi/matplotlib
export XDG_CACHE_HOME=/lustre/hdd/LAS/jannesar-lab/arushi/cache
export TRITON_CACHE_DIR=/lustre/hdd/LAS/jannesar-lab/arushi/cache/triton
export HF_HOME=/lustre/hdd/LAS/jannesar-lab/arushi/
export HF_CACHE=/lustre/hdd/LAS/jannesar-lab/arushi/activation-steering/models
export TRANSFORMERS_CACHE=/lustre/hdd/LAS/jannesar-lab/arushi/activation-steering/models

cd /lustre/hdd/LAS/jannesar-lab/arushi
source myenv/bin/activate

export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

cd activation-steering/

########################################
# CLI args
########################################
# Usage examples:
#   sbatch slurm/run_mcq_build.sh --dataset manysstubs --fewshot-k 3 --seed 42
#   sbatch slurm/run_mcq_build.sh --dataset tssb       --fewshot-k 0 --seed 123
########################################

DATASET="manysstubs"   # tssb | manysstubs
FEWSHOT_K=3
SEED=42

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --fewshot-k)
      FEWSHOT_K="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

ROOT="/lustre/hdd/LAS/jannesar-lab/arushi/activation-steering"

case "$DATASET" in
  tssb)
    SOURCE_JSON="${ROOT}/data/tssb_data_3M/processed/tssb_eval_rest.jsonl"
    ;;
  manysstubs)
    SOURCE_JSON="${ROOT}/data/manysstubs4j/processed/sstubs_eval_rest.jsonl"
    ;;
  *)
    echo "ERROR: --dataset must be 'tssb' or 'manysstubs' (got '$DATASET')."
    exit 1
    ;;
esac

mkdir -p mcq_cache

MCQ_OUT="mcq_cache/${DATASET}_mcq_k${FEWSHOT_K}_seed${SEED}.json"

echo "============================================"
echo "Building MCQ questions"
echo "  Dataset   : ${DATASET}"
echo "  Source    : ${SOURCE_JSON}"
echo "  Few-shot k: ${FEWSHOT_K}"
echo "  Seed      : ${SEED}"
echo "  Output    : ${MCQ_OUT}"
echo "============================================"

python scripts/ab_apr_eval.py \
  --source_dataset "${SOURCE_JSON}" \
  --output_mcq_questions "${MCQ_OUT}" \
  --fewshot_k "${FEWSHOT_K}" \
  --seed "${SEED}" \
  --build_only

echo "[INFO] Done building MCQ questions."

