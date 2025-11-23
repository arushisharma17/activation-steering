#!/bin/bash

#SBATCH --time=3-05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --gres=gpu:a100
#SBATCH --mem=128G
#SBATCH --job-name="apr steering"
#SBATCH --mail-user=arushi17@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="logs/slurm-%j.out"

set -euo pipefail

########################################
# Environment setup (Nova)
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
# Config
########################################

ROOT="/lustre/hdd/LAS/jannesar-lab/arushi/activation-steering"
PYTHON="python"
TRAIN_SCRIPT="${ROOT}/scripts/train_behavior_vector.py"

# Default: run nothing unless datasets are chosen
DATASET="none"

# Default dataset JSON paths (auto-filled once dataset is chosen)
TSSB_JSON="${ROOT}/data/tssb_data_3M/derived/correctness_behavior_apr.json"
MANY_JSON="${ROOT}/data/manysstubs4j/processed/apr_manysstubs_pairs_3k.json"

# Selected dataset path (filled later)
PAIRS_JSON=""

MAX_EXAMPLES=300
HF_CACHE="/lustre/hdd/LAS/jannesar-lab/arushi/activation-steering/models"
METHOD="pca_center"
LAST_TOKENS="suffix-only"

# Models (if none chosen → run all)
RUN_CODELLAMA=false
RUN_QWEN_INST_7B=false
RUN_QWEN_CODER_7B=false
RUN_QWEN_CODER_14B=false

########################################
# Usage
########################################

usage() {
  cat <<EOF

Usage: $0 --dataset {tssb|manysstubs|both} [model-flags] [options]

DATASET OPTIONS (required):
  --dataset tssb           Use TSSB pairs JSON
  --dataset manysstubs     Use ManySStuBs Java pairs JSON
  --dataset both           Run both datasets sequentially

MODEL FLAGS (optional; if none given: run ALL):
  --codellama              Run CodeLlama-7B-Instruct
  --qwen-inst-7b           Run Qwen2.5-7B-Instruct
  --qwen-coder-7b          Run Qwen2.5-Coder-7B-Instruct
  --qwen-coder-14b         Run Qwen2.5-Coder-14B-Instruct

OTHER OPTIONS:
  --max-examples N         Default: 300
  --hf-cache DIR           HuggingFace cache (default: ${HF_CACHE})
  --method METHOD          Default: pca_center
  --last-tokens VALUE      Default: suffix-only
  --pairs-json PATH        Override dataset JSON manually

EXAMPLES:

# Run all 4 models on ManySStuBs
  $0 --dataset manysstubs

# Run tssb on Qwen Coder 7B ONLY
  $0 --dataset tssb --qwen-coder-7b

# Run both datasets and all models
  $0 --dataset both

EOF
  exit 1
}

########################################
# Parse args
########################################

if [[ $# -eq 0 ]]; then usage; fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"; shift 2;;
    --pairs-json)
      PAIRS_JSON="$2"; shift 2;;

    --max-examples)
      MAX_EXAMPLES="$2"; shift 2;;
    --hf-cache)
      HF_CACHE="$2"; shift 2;;
    --method)
      METHOD="$2"; shift 2;;
    --last-tokens)
      LAST_TOKENS="$2"; shift 2;;

    --codellama)
      RUN_CODELLAMA=true; shift;;
    --qwen-inst-7b)
      RUN_QWEN_INST_7B=true; shift;;
    --qwen-coder-7b)
      RUN_QWEN_CODER_7B=true; shift;;
    --qwen-coder-14b)
      RUN_QWEN_CODER_14B=true; shift;;

    -h|--help)
      usage;;
    *)
      echo "Unknown argument: $1"; usage;;
  esac
done

########################################
# Validate dataset choice
########################################

if [[ "${DATASET}" == "none" ]]; then
  echo "[ERROR] --dataset flag is required."
  usage
fi

########################################
# Helper to run one model
########################################

run_for_model() {
  local JSON_PATH="$1"
  local DATASET_NAME="$2"
  local MODEL_ID="$3"
  local TOKENIZER_ID="$4"
  local TAG="$5"

  echo
  echo "======================================================="
  echo ">>> Training vector: Dataset=${DATASET_NAME}, Model=${TAG}"
  echo "    JSON: ${JSON_PATH}"
  echo "======================================================="

  HF_ARG=""
  if [[ -n "${HF_CACHE}" ]]; then
    HF_ARG="--hf-cache ${HF_CACHE}"
  fi

  ${PYTHON} "${TRAIN_SCRIPT}" \
    --pairs-json "${JSON_PATH}" \
    --dataset-name "${DATASET_NAME}" \
    --max-examples "${MAX_EXAMPLES}" \
    --model-id "${MODEL_ID}" \
    --tokenizer-id "${TOKENIZER_ID}" \
    --method "${METHOD}" \
    --last-tokens "${LAST_TOKENS}" \
    ${HF_ARG}
}

########################################
# Helper: Run all selected models for a dataset
########################################

run_models_for_dataset() {
  local JSON_PATH="$1"
  local DATASET_NAME="$2"

  # If no model flags chosen → run all
  if ! $RUN_CODELLAMA && ! $RUN_QWEN_INST_7B && ! $RUN_QWEN_CODER_7B && ! $RUN_QWEN_CODER_14B; then
    RUN_CODELLAMA=true
    RUN_QWEN_INST_7B=true
    RUN_QWEN_CODER_7B=true
    RUN_QWEN_CODER_14B=true
  fi

  if $RUN_CODELLAMA; then
    run_for_model "$JSON_PATH" "$DATASET_NAME" \
      "meta-llama/CodeLlama-7b-Instruct-hf" \
      "meta-llama/CodeLlama-7b-Instruct-hf" \
      "CodeLlama-7B"
  fi

  if $RUN_QWEN_INST_7B; then
    run_for_model "$JSON_PATH" "$DATASET_NAME" \
      "Qwen/Qwen2.5-7B-Instruct" \
      "Qwen/Qwen2.5-7B-Instruct" \
      "Qwen2.5-7B-Instruct"
  fi

  if $RUN_QWEN_CODER_7B; then
    run_for_model "$JSON_PATH" "$DATASET_NAME" \
      "Qwen/Qwen2.5-Coder-7B-Instruct" \
      "Qwen/Qwen2.5-Coder-7B-Instruct" \
      "Qwen2.5-Coder-7B"
  fi

  if $RUN_QWEN_CODER_14B; then
    run_for_model "$JSON_PATH" "$DATASET_NAME" \
      "Qwen/Qwen2.5-Coder-14B-Instruct" \
      "Qwen/Qwen2.5-Coder-14B-Instruct" \
      "Qwen2.5-Coder-14B"
  fi
}

########################################
# Execute based on dataset selection
########################################

if [[ -n "${PAIRS_JSON}" ]]; then
  # User overrides dataset path manually
  echo "[INFO] Using manually specified pairs JSON: ${PAIRS_JSON}"
  run_models_for_dataset "${PAIRS_JSON}" "custom"
  exit 0
fi

case "${DATASET}" in
  tssb)
    run_models_for_dataset "${TSSB_JSON}" "tssb"
    ;;
  manysstubs)
    run_models_for_dataset "${MANY_JSON}" "manysstubs"
    ;;
  both)
    run_models_for_dataset "${TSSB_JSON}" "tssb"
    run_models_for_dataset "${MANY_JSON}" "manysstubs"
    ;;
  *)
    echo "[ERROR] Unknown dataset: ${DATASET}"
    usage;;
esac

echo "All requested steering vectors completed."

