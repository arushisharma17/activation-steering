#!/bin/bash

#SBATCH --time=1-05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --gres=gpu:a100
#SBATCH --mem=128G
#SBATCH --job-name="apr_ab_eval"
#SBATCH --mail-user=arushi17@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="logs/slurm-%j.out"

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

########################################
# Parse script arguments
########################################
# Usage examples:
#   sbatch run_ab_eval.sh \
#       --dataset manysstubs \
#       --pairs-path /lustre/.../activation-steering/data/manysstubs4j/processed/sstubs_eval_rest.jsonl
#
#   sbatch run_ab_eval.sh \
#       --dataset tssb \
#       --pairs-path /lustre/.../activation-steering/data/tssb_data_3M/processed/tssb_eval_rest.jsonl \
#       --qwen-coder-7b --qwen-coder-14b
#
########################################

DATASET="manysstubs"       # tssb | manysstubs
PAIRS_PATH=""              # must be provided with --pairs-path
MODE="compare"             # compare | baseline | steered
LAYERS="last:4"
STRENGTH="2.0"

# model selection flags
RUN_CODELLAMA=0
RUN_QWEN_INST_7B=0
RUN_QWEN_CODER_7B=0
RUN_QWEN_CODER_14B=0
EXPLICIT_MODELS=0          # track whether user chose models

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --pairs-path)
      PAIRS_PATH="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"           # baseline | steered | compare
      shift 2
      ;;
    --layers)
      LAYERS="$2"         # e.g. "last:4" or "all" or "27,28,29,30,31"
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

if [[ -z "$PAIRS_PATH" ]]; then
  echo "ERROR: --pairs-path is required (JSONL with before/after)."
  exit 1
fi

# If user didn't explicitly pick models, run all four
if [[ "$EXPLICIT_MODELS" -eq 0 ]]; then
  RUN_CODELLAMA=1
  RUN_QWEN_INST_7B=1
  RUN_QWEN_CODER_7B=1
  RUN_QWEN_CODER_14B=1
fi

########################################
# Dataset-specific naming (for vectors)
########################################
# Assumes you trained steering vectors with train_behavior_vector.py using:
#   --dataset-name tssb       and pairs-json .../data/tssb_data_3M/processed/apr_tssb_pairs_3k.json
#   --dataset-name manysstubs and pairs-json .../data/manysstubs4j/processed/apr_manysstubs_pairs_3k.json
#
# So vector filenames follow:
#   vectors/java/behavior_<dataset>_<pairs_stem_slug>_<model_slug>_pca_center_suffix.svec
#
########################################

VECTOR_DIR="vectors/java"
VECTOR_SUFFIX="pca_center_suffix"

case "$DATASET" in
  tssb)
    DATASET_SLUG="tssb"
    PAIRS_STEM_SLUG="apr_tssb_pairs_3k"
    ;;
  manysstubs)
    DATASET_SLUG="manysstubs"
    PAIRS_STEM_SLUG="apr_manysstubs_pairs_3k"
    ;;
  *)
    echo "ERROR: --dataset must be 'tssb' or 'manysstubs' (got '$DATASET')."
    exit 1
    ;;
esac

########################################
# Model IDs and slugs (match train_behavior_vector.py naming)
########################################

# 1) CodeLlama-7B-Instruct
CODELLAMA_MODEL="meta-llama/CodeLlama-7b-Instruct-hf"
CODELLAMA_TOKENIZER="$CODELLAMA_MODEL"
CODELLAMA_SLUG="codellama-7b-instruct-hf"

# 2) Qwen2.5-7B-Instruct
QWEN_INST_7B_MODEL="Qwen/Qwen2.5-7B-Instruct"
QWEN_INST_7B_TOKENIZER="$QWEN_INST_7B_MODEL"
QWEN_INST_7B_SLUG="qwen2-5-7b-instruct"

# 3) Qwen2.5-Coder-7B-Instruct
QWEN_CODER_7B_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
QWEN_CODER_7B_TOKENIZER="$QWEN_CODER_7B_MODEL"
QWEN_CODER_7B_SLUG="qwen2-5-coder-7b-instruct"

# 4) Qwen2.5-Coder-14B-Instruct
QWEN_CODER_14B_MODEL="Qwen/Qwen2.5-Coder-14B-Instruct"
QWEN_CODER_14B_TOKENIZER="$QWEN_CODER_14B_MODEL"
QWEN_CODER_14B_SLUG="qwen2-5-coder-14b-instruct"

########################################
# Shared eval settings
########################################

FEWSHOT_K=3
SEED=42
SHOW_N=6

# Build common mode flags:
MODE_ARGS=()
case "$MODE" in
  compare)
    MODE_ARGS+=(--compare)
    ;;
  baseline)
    MODE_ARGS+=(--mode baseline)
    ;;
  steered)
    MODE_ARGS+=(--mode steered)
    ;;
  *)
    echo "ERROR: --mode must be one of: compare | baseline | steered"
    exit 1
    ;;
esac

########################################
# Helper function to run one model
########################################

run_eval () {
  local model_id="$1"
  local tokenizer_id="$2"
  local model_slug="$3"

  local vec_path="${VECTOR_DIR}/behavior_${DATASET_SLUG}_${PAIRS_STEM_SLUG}_${model_slug}_${VECTOR_SUFFIX}.svec"

  echo "======================================================"
  echo "Running A/B APR eval"
  echo "  Dataset     : ${DATASET_SLUG}"
  echo "  Pairs (eval): ${PAIRS_PATH}"
  echo "  Model       : ${model_id}"
  echo "  Tokenizer   : ${tokenizer_id}"
  echo "  Vector      : ${vec_path}"
  echo "  Mode        : ${MODE}"
  echo "  Layers      : ${LAYERS}"
  echo "  Strength    : ${STRENGTH}"
  echo "======================================================"

  python scripts/ab_apr_eval.py \
    --pairs_path "${PAIRS_PATH}" \
    --model_id "${model_id}" \
    --tokenizer_id "${tokenizer_id}" \
    --vector_path "${vec_path}" \
    --layers "${LAYERS}" \
    --strength "${STRENGTH}" \
    --fewshot_k "${FEWSHOT_K}" \
    --seed "${SEED}" \
    --show_n "${SHOW_N}" \
    "${MODE_ARGS[@]}"
}

########################################
# Run selected models
########################################

if [[ "$RUN_CODELLAMA" -eq 1 ]]; then
  run_eval "${CODELLAMA_MODEL}" "${CODELLAMA_TOKENIZER}" "${CODELLAMA_SLUG}"
fi

if [[ "$RUN_QWEN_INST_7B" -eq 1 ]]; then
  run_eval "${QWEN_INST_7B_MODEL}" "${QWEN_INST_7B_TOKENIZER}" "${QWEN_INST_7B_SLUG}"
fi

if [[ "$RUN_QWEN_CODER_7B" -eq 1 ]]; then
  run_eval "${QWEN_CODER_7B_MODEL}" "${QWEN_CODER_7B_TOKENIZER}" "${QWEN_CODER_7B_SLUG}"
fi

if [[ "$RUN_QWEN_CODER_14B" -eq 1 ]]; then
  run_eval "${QWEN_CODER_14B_MODEL}" "${QWEN_CODER_14B_TOKENIZER}" "${QWEN_CODER_14B_SLUG}"
fi

