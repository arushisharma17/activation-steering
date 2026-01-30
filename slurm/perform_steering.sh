#!/bin/bash
set -euo pipefail

########################################
# Portable Environment Setup (ANON)
########################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# Optional virtualenv activation (generic)
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
# Parse script arguments
########################################

DATASET="tssb"          # only supported option in anonymous artifact
MODE="compare"          # baseline | steered | compare
FEWSHOT_K=0
SEED=42
LAYERS="last:4"
STRENGTH="2.0"
MCQ_FILE=""             # if empty, use default mcq_cache/<dataset>_mcq_kK_seedS.json

# subset controls (pass-through to ab_apr_eval.py)
EVAL_START=0
EVAL_LIMIT=0            # 0 = all remaining

# model selection flags
RUN_CODELLAMA=0
RUN_QWEN_INST_7B=0
RUN_QWEN_CODER_7B=0
RUN_QWEN_CODER_14B=0
EXPLICIT_MODELS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"; shift 2;;
    --mode)
      MODE="$2"; shift 2;;
    --fewshot-k)
      FEWSHOT_K="$2"; shift 2;;
    --seed)
      SEED="$2"; shift 2;;
    --layers)
      LAYERS="$2"; shift 2;;
    --strength)
      STRENGTH="$2"; shift 2;;
    --mcq-file)
      MCQ_FILE="$2"; shift 2;;
    --eval-start)
      EVAL_START="$2"; shift 2;;
    --eval-limit)
      EVAL_LIMIT="$2"; shift 2;;

    --codellama-7b)
      RUN_CODELLAMA=1; EXPLICIT_MODELS=1; shift;;
    --qwen-inst-7b)
      RUN_QWEN_INST_7B=1; EXPLICIT_MODELS=1; shift;;
    --qwen-coder-7b)
      RUN_QWEN_CODER_7B=1; EXPLICIT_MODELS=1; shift;;
    --qwen-coder-14b)
      RUN_QWEN_CODER_14B=1; EXPLICIT_MODELS=1; shift;;

    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Enforce TSSB-only
if [[ "${DATASET}" != "tssb" ]]; then
  echo "[ERROR] This anonymous artifact supports only --dataset tssb."
  exit 1
fi

# Default MCQ file if not specified
if [[ -z "${MCQ_FILE}" ]]; then
  MCQ_FILE="mcq_cache/${DATASET}_mcq_k${FEWSHOT_K}_seed${SEED}.json"
fi

if [[ ! -f "${MCQ_FILE}" ]]; then
  echo "[ERROR] MCQ file not found: ${MCQ_FILE}"
  echo "        Build it first with your MCQ builder (or whatever generates MCQ JSON)."
  exit 1
fi

# If user didn't explicitly pick models, run all four
if [[ "${EXPLICIT_MODELS}" -eq 0 ]]; then
  RUN_CODELLAMA=1
  RUN_QWEN_INST_7B=1
  RUN_QWEN_CODER_7B=1
  RUN_QWEN_CODER_14B=1
fi

########################################
# TSSB-only vector naming
########################################

DATASET_SLUG="tssb"
VECTOR_ROOT="vectors/python"
VECTOR_PREFIX="correctness_vector_100"
VECTOR_SUFFIX="pca_center_suffix"

########################################
# Model IDs and slugs (match vector naming)
########################################

CODELLAMA_MODEL="meta-llama/CodeLlama-7b-Instruct-hf"
CODELLAMA_TOKENIZER="${CODELLAMA_MODEL}"
CODELLAMA_SLUG="codellama-7b-instruct-hf"

QWEN_INST_7B_MODEL="Qwen/Qwen2.5-7B-Instruct"
QWEN_INST_7B_TOKENIZER="${QWEN_INST_7B_MODEL}"
QWEN_INST_7B_SLUG="qwen2-5-7b-instruct"

QWEN_CODER_7B_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
QWEN_CODER_7B_TOKENIZER="${QWEN_CODER_7B_MODEL}"
QWEN_CODER_7B_SLUG="qwen2-5-coder-7b-instruct"

QWEN_CODER_14B_MODEL="Qwen/Qwen2.5-Coder-14B-Instruct"
QWEN_CODER_14B_TOKENIZER="${QWEN_CODER_14B_MODEL}"
QWEN_CODER_14B_SLUG="qwen2-5-coder-14b-instruct"

########################################
# Shared eval settings
########################################

SHOW_N=6

MODE_ARGS=()
case "${MODE}" in
  compare)  MODE_ARGS+=(--compare) ;;
  baseline) MODE_ARGS+=(--mode baseline) ;;
  steered)  MODE_ARGS+=(--mode steered) ;;
  *)
    echo "[ERROR] --mode must be one of: compare | baseline | steered"
    exit 1
    ;;
esac

mkdir -p mcq_cache

########################################
# Helper function to run one model
########################################

run_eval () {
  local model_id="$1"
  local tokenizer_id="$2"
  local model_slug="$3"

  local vec_path="${VECTOR_ROOT}/${VECTOR_PREFIX}_${model_slug}_${VECTOR_SUFFIX}.svec"

  local baseline_cache="mcq_cache/baseline_${DATASET_SLUG}_${model_slug}_k${FEWSHOT_K}_seed${SEED}_es${EVAL_START}_el${EVAL_LIMIT}.json"

  echo "======================================================"
  echo "Running A/B APR MCQ eval"
  echo "  Dataset       : ${DATASET_SLUG}"
  echo "  MCQ file      : ${MCQ_FILE}"
  echo "  Model         : ${model_id}"
  echo "  Vector        : ${vec_path}"
  echo "  Mode          : ${MODE}"
  echo "  Few-shot k    : ${FEWSHOT_K}"
  echo "  Seed          : ${SEED}"
  echo "  Layers        : ${LAYERS}"
  echo "  Strength      : ${STRENGTH}"
  echo "  Eval start    : ${EVAL_START}"
  echo "  Eval limit    : ${EVAL_LIMIT}"
  echo "  Baseline cache: ${baseline_cache}"
  echo "======================================================"

  python scripts/ab_apr_eval.py \
    --mcq_questions "${MCQ_FILE}" \
    --model_id "${model_id}" \
    --tokenizer_id "${tokenizer_id}" \
    --vector_path "${vec_path}" \
    --layers "${LAYERS}" \
    --strength "${STRENGTH}" \
    --fewshot_k "${FEWSHOT_K}" \
    --seed "${SEED}" \
    --show_n "${SHOW_N}" \
    --baseline_cache "${baseline_cache}" \
    --eval_start "${EVAL_START}" \
    --eval_limit "${EVAL_LIMIT}" \
    "${MODE_ARGS[@]}"
}

########################################
# Run selected models
########################################

if [[ "${RUN_CODELLAMA}" -eq 1 ]]; then
  run_eval "${CODELLAMA_MODEL}" "${CODELLAMA_TOKENIZER}" "${CODELLAMA_SLUG}"
fi

if [[ "${RUN_QWEN_INST_7B}" -eq 1 ]]; then
  run_eval "${QWEN_INST_7B_MODEL}" "${QWEN_INST_7B_TOKENIZER}" "${QWEN_INST_7B_SLUG}"
fi

if [[ "${RUN_QWEN_CODER_7B}" -eq 1 ]]; then
  run_eval "${QWEN_CODER_7B_MODEL}" "${QWEN_CODER_7B_TOKENIZER}" "${QWEN_CODER_7B_SLUG}"
fi

if [[ "${RUN_QWEN_CODER_14B}" -eq 1 ]]; then
  run_eval "${QWEN_CODER_14B_MODEL}" "${QWEN_CODER_14B_TOKENIZER}" "${QWEN_CODER_14B_SLUG}"
fi

