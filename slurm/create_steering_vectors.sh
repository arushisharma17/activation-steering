#!/bin/bash
set -euo pipefail

########################################
# Portable Environment Setup (ANON)
########################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

# Optional virtualenv activation
if [[ -f "${ROOT_DIR}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv/bin/activate"
elif [[ -f "${ROOT_DIR}/venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/venv/bin/activate"
else
  echo "[WARN] No virtual environment found — relying on system Python."
fi

########################################
# Config
########################################

ROOT="${ROOT_DIR}"
PYTHON="python"
TRAIN_SCRIPT="${ROOT}/scripts/train_behavior_vector.py"

# Dataset selection
DATASET="none"

# TSSB only
TSSB_JSON="${ROOT}/data/tssb_data_3M/derived/correctness_behavior_apr.json"

# Override path
PAIRS_JSON=""

MAX_EXAMPLES=200
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

Usage: $0 --dataset tssb [model-flags] [options]

DATASET OPTIONS:
  --dataset tssb           Use TSSB pairs JSON (default option)

MODEL FLAGS (optional; if none given: run ALL):
  --codellama              Run CodeLlama-7B-Instruct
  --qwen-inst-7b           Run Qwen2.5-7B-Instruct
  --qwen-coder-7b          Run Qwen2.5-Coder-7B-Instruct
  --qwen-coder-14b         Run Qwen2.5-Coder-14B-Instruct

OTHER OPTIONS:
  --max-examples N         Default: 200
  --method METHOD          Default: pca_center
  --last-tokens VALUE      Default: suffix-only
  --pairs-json PATH        Override dataset JSON manually

EXAMPLES:

# Run all models on TSSB
  $0 --dataset tssb

# Run only Qwen Coder 7B
  $0 --dataset tssb --qwen-coder-7b

# Run custom dataset
  $0 --pairs-json /path/to/custom_pairs.json --dataset tssb

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

if [[ -z "${PAIRS_JSON}" && "${DATASET}" != "tssb" ]]; then
  echo "[ERROR] Only TSSB is supported in the anonymous artifact."
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

  ${PYTHON} "${TRAIN_SCRIPT}" \
    --pairs-json "${JSON_PATH}" \
    --dataset-name "${DATASET_NAME}" \
    --max-examples "${MAX_EXAMPLES}" \
    --model-id "${MODEL_ID}" \
    --tokenizer-id "${TOKENIZER_ID}" \
    --method "${METHOD}" \
    --last-tokens "${LAST_TOKENS}"
}

########################################
# Helper: Run all selected models
########################################

run_models() {
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
# Execute
########################################

if [[ -n "${PAIRS_JSON}" ]]; then
  echo "[INFO] Using manually specified pairs JSON: ${PAIRS_JSON}"
  run_models "${PAIRS_JSON}" "custom"
else
  run_models "${TSSB_JSON}" "tssb"
fi

echo "All requested steering vectors completed."

