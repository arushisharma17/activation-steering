#!/bin/bash

#SBATCH --time=1-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --job-name="mcq_fewshot_sweep"
#SBATCH --mail-user=arushi17@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="logs/slurm-mcq-fsweep-%j.out"

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

nvidia-smi

export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

cd activation-steering/

mkdir -p mcq_cache

########################################
# CLI args
########################################
# Usage examples:
#
#   # Sweep k in {0,1,3,5} for all models on ManySStuBs
#   sbatch slurm/run_mcq_fewshot_sweep.sh \
#     --dataset manysstubs \
#     --seed 42
#
#   # Sweep only k=0,3 for Qwen coder 7B on TSSB
#   sbatch slurm/run_mcq_fewshot_sweep.sh \
#     --dataset tssb \
#     --seed 42 \
#     --k-list "0 3" \
#     --qwen-coder-7b
########################################

DATASET="manysstubs"     # tssb | manysstubs
SEED=42
K_LIST="0 1 3 5"         # space-separated list of few-shot k values
LAYERS="last:4"          # not used here (baseline only) but logged for completeness
STRENGTH="2.0"           # same as above

# model selection flags
RUN_CODELLAMA=0
RUN_QWEN_INST_7B=0
RUN_QWEN_CODER_7B=0
RUN_QWEN_CODER_14B=0
EXPLICIT_MODELS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --k-list)
      K_LIST="$2"
      shift 2
      ;;
    --layers)
      LAYERS="$2"
      shift 2
      ;;
    --strength)
      STRENGTH="$2"
      shift 2
      ;;
    # model selection
    --codellama-7b)
      RUN_CODELLAMA=1
      EXPLICIT_MODELS=1
      shift
      ;;
    --qwen-inst-7b)
      RUN_QWEN_INST_7B=1
      EXPLICIT_MODELS=1
      shift
      ;;
    --qwen-coder-7b)
      RUN_QWEN_CODER_7B=1
      EXPLICIT_MODELS=1
      shift
      ;;
    --qwen-coder-14b)
      RUN_QWEN_CODER_14B=1
      EXPLICIT_MODELS=1
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# If user didn’t explicitly pick models, run all four
if [[ "$EXPLICIT_MODELS" -eq 0 ]]; then
  RUN_CODELLAMA=1
  RUN_QWEN_INST_7B=1
  RUN_QWEN_CODER_7B=1
  RUN_QWEN_CODER_14B=1
fi

ROOT="/lustre/hdd/LAS/jannesar-lab/arushi/activation-steering"

########################################
# Dataset -> source JSON (for building MCQs)
########################################

case "$DATASET" in
  tssb)
    SOURCE_JSON="${ROOT}/data/tssb_data_3M/derived/filtered-2.jsonl"
    DATASET_SLUG="tssb"
    PAIRS_STEM_SLUG="apr_tssb_pairs_3k"
    ;;
  manysstubs)
    SOURCE_JSON="${ROOT}/data/manysstubs4j/processed/sstubs_eval_holdout.jsonl"
    DATASET_SLUG="manysstubs"
    PAIRS_STEM_SLUG="apr_manysstubs_pairs_3k"
    ;;
  *)
    echo "ERROR: --dataset must be 'tssb' or 'manysstubs' (got '$DATASET')."
    exit 1
    ;;
esac

########################################
# Model IDs and slugs (match vector naming)
########################################

CODELLAMA_MODEL="meta-llama/CodeLlama-7b-Instruct-hf"
CODELLAMA_TOKENIZER="$CODELLAMA_MODEL"
CODELLAMA_SLUG="codellama-7b-instruct-hf"

QWEN_INST_7B_MODEL="Qwen/Qwen2.5-7B-Instruct"
QWEN_INST_7B_TOKENIZER="$QWEN_INST_7B_MODEL"
QWEN_INST_7B_SLUG="qwen2-5-7b-instruct"

QWEN_CODER_7B_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
QWEN_CODER_7B_TOKENIZER="$QWEN_CODER_7B_MODEL"
QWEN_CODER_7B_SLUG="qwen2-5-coder-7b-instruct"

QWEN_CODER_14B_MODEL="Qwen/Qwen2.5-Coder-14B-Instruct"
QWEN_CODER_14B_TOKENIZER="$QWEN_CODER_14B_MODEL"
QWEN_CODER_14B_SLUG="qwen2-5-coder-14b-instruct"

########################################
# Helper: build MCQ file for a given k
########################################

build_mcq_if_needed () {
  local k="$1"
  local mcq_file="mcq_cache/${DATASET_SLUG}_mcq_k${k}_seed${SEED}.json"

  if [[ -f "$mcq_file" ]]; then
    echo "[INFO] MCQ file exists for k=${k}: ${mcq_file}"
  else
    echo "============================================"
    echo "[INFO] Building MCQ questions for k=${k}"
    echo "  Dataset : ${DATASET_SLUG}"
    echo "  Source  : ${SOURCE_JSON}"
    echo "  Seed    : ${SEED}"
    echo "  Output  : ${mcq_file}"
    echo "============================================"

    python scripts/ab_apr_eval.py \
      --source_dataset "${SOURCE_JSON}" \
      --output_mcq_questions "${mcq_file}" \
      --fewshot_k "${k}" \
      --seed "${SEED}" \
      --build_only
  fi
}

########################################
# Helper: run baseline eval for one model, one k
########################################

run_baseline_for_model_k () {
  local model_id="$1"
  local tokenizer_id="$2"
  local model_slug="$3"
  local k="$4"

  local mcq_file="mcq_cache/${DATASET_SLUG}_mcq_k${k}_seed${SEED}.json"
  local baseline_cache="mcq_cache/baseline_${DATASET_SLUG}_${model_slug}_k${k}_seed${SEED}.json"

  echo "======================================================"
  echo "Baseline A/B MCQ eval"
  echo "  Dataset      : ${DATASET_SLUG}"
  echo "  k (few-shot) : ${k}"
  echo "  MCQ file     : ${mcq_file}"
  echo "  Model        : ${model_id}"
  echo "  Tokenizer    : ${tokenizer_id}"
  echo "  Seed         : ${SEED}"
  echo "  Baseline cache: ${baseline_cache}"
  echo "======================================================"

  python scripts/ab_apr_eval.py \
    --mcq_questions "${mcq_file}" \
    --model_id "${model_id}" \
    --tokenizer_id "${tokenizer_id}" \
    --mode baseline \
    --fewshot_k "${k}" \
    --seed "${SEED}" \
    --baseline_cache "${baseline_cache}" \
    --show_n 6
}

########################################
# Sweep over k and models
########################################

echo "[INFO] Few-shot sweep for dataset=${DATASET_SLUG}, seed=${SEED}"
echo "[INFO] k-list: ${K_LIST}"

for K in ${K_LIST}; do
  build_mcq_if_needed "${K}"

  if [[ "$RUN_CODELLAMA" -eq 1 ]]; then
    run_baseline_for_model_k "${CODELLAMA_MODEL}" "${CODELLAMA_TOKENIZER}" "${CODELLAMA_SLUG}" "${K}"
  fi

  if [[ "$RUN_QWEN_INST_7B" -eq 1 ]]; then
    run_baseline_for_model_k "${QWEN_INST_7B_MODEL}" "${QWEN_INST_7B_TOKENIZER}" "${QWEN_INST_7B_SLUG}" "${K}"
  fi

  if [[ "$RUN_QWEN_CODER_7B" -eq 1 ]]; then
    run_baseline_for_model_k "${QWEN_CODER_7B_MODEL}" "${QWEN_CODER_7B_TOKENIZER}" "${QWEN_CODER_7B_SLUG}" "${K}"
  fi

  if [[ "$RUN_QWEN_CODER_14B" -eq 1 ]]; then
    run_baseline_for_model_k "${QWEN_CODER_14B_MODEL}" "${QWEN_CODER_14B_TOKENIZER}" "${QWEN_CODER_14B_SLUG}" "${K}"
  fi
done

echo "[INFO] Few-shot sweep completed."

