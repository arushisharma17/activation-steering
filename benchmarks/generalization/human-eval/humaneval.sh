#!/usr/bin/env bash
#
# HumanEval multi-model runner with optional steering using APR behavior vectors
# (portable, anonymized for double-blind submission)
#

set -euo pipefail

########################################
# Parse CLI arguments
########################################

RUN_BASELINE=false
RUN_STEERED=false
EVAL_ONLY=false
BEHAVIOR_DATASET="default"
N_SAMPLES=10

RUN_QWEN_INST_7B=false
RUN_QWEN_CODER_7B=false
RUN_QWEN_CODER_14B=false
RUN_CODELLAMA_7B=false
ANY_MODEL_FLAG=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --baseline) RUN_BASELINE=true; shift ;;
    --steered)  RUN_STEERED=true; shift ;;
    --eval-only) EVAL_ONLY=true; shift ;;
    --behavior-dataset) BEHAVIOR_DATASET="$2"; shift 2 ;;
    --n) N_SAMPLES="$2"; shift 2 ;;
    --qwen-inst-7b) RUN_QWEN_INST_7B=true; ANY_MODEL_FLAG=true; shift ;;
    --qwen-coder-7b) RUN_QWEN_CODER_7B=true; ANY_MODEL_FLAG=true; shift ;;
    --qwen-coder-14b) RUN_QWEN_CODER_14B=true; ANY_MODEL_FLAG=true; shift ;;
    --codellama-7b) RUN_CODELLAMA_7B=true; ANY_MODEL_FLAG=true; shift ;;
    *)
      echo "[ERROR] Unknown option: $1"
      exit 1
      ;;
  esac
done

if [ "$RUN_BASELINE" = false ] && [ "$RUN_STEERED" = false ] && [ "$EVAL_ONLY" = false ]; then
  RUN_BASELINE=true
  RUN_STEERED=true
fi

if [ "$ANY_MODEL_FLAG" = false ]; then
  RUN_QWEN_INST_7B=true
  RUN_QWEN_CODER_7B=true
  RUN_QWEN_CODER_14B=true
  RUN_CODELLAMA_7B=true
fi

########################################
# Minimal HuggingFace configuration (optional)
########################################

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"

########################################
# Project paths (system-agnostic)
########################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

HE_ROOT="${PROJECT_ROOT}/benchmarks/generalization/human-eval"
VEC_ROOT="${PROJECT_ROOT}/vectors"

if [[ ! -d "$HE_ROOT" ]]; then
  echo "[ERROR] HE_ROOT not found: $HE_ROOT"
  exit 1
fi

cd "$HE_ROOT"
mkdir -p logs results

########################################
# Logging
########################################

echo "======================================================="
echo "[CONFIG]"
echo "PROJECT_ROOT     = $PROJECT_ROOT"
echo "RUN_BASELINE     = $RUN_BASELINE"
echo "RUN_STEERED      = $RUN_STEERED"
echo "EVAL_ONLY        = $EVAL_ONLY"
echo "N_SAMPLES        = $N_SAMPLES"
echo "======================================================="

########################################
# Model Config
########################################

MODELS=(
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-Coder-7B-Instruct"
  "Qwen/Qwen2.5-Coder-14B-Instruct"
  "meta-llama/CodeLlama-7b-Instruct-hf"
)

SLUGS=(
  "Qwen2.5-7B-Instruct"
  "Qwen2.5-Coder-7B-Instruct"
  "Qwen2.5-Coder-14B-Instruct"
  "CodeLlama-7b-Instruct-hf"
)

########################################
# Placeholder vector locations
########################################
# Expected layout:
#   $VEC_ROOT/python/Qwen2.5-7B-Instruct.svec
#   $VEC_ROOT/java/Qwen2.5-7B-Instruct.svec
#
# (actual mapping documented separately in artifact README)

vector_path_for_model() {
  local slug="$1"
  local lang="$2"   # python | java

  echo "${VEC_ROOT}/${lang}/${slug}.svec"
}

should_run_model_idx() {
  local idx=$1
  case $idx in
    0) $RUN_QWEN_INST_7B && return 0 ;;
    1) $RUN_QWEN_CODER_7B && return 0 ;;
    2) $RUN_QWEN_CODER_14B && return 0 ;;
    3) $RUN_CODELLAMA_7B && return 0 ;;
  esac
  return 1
}

########################################
# BASELINE RUNS
########################################

if [ "$EVAL_ONLY" = false ] && [ "$RUN_BASELINE" = true ]; then
  echo "[INFO] === BASELINE RUNS ==="

  for idx in 0 1 2 3; do
    if ! should_run_model_idx "$idx"; then
      continue
    fi

    MODEL_ID="${MODELS[$idx]}"
    echo "[BASELINE] $MODEL_ID"

    python gen.py \
      --model "$MODEL_ID" \
      --n "$N_SAMPLES" \
      --dataset-tag "none" \
      --condition "baseline"
  done
fi

########################################
# STEERED RUNS
########################################

if [ "$EVAL_ONLY" = false ] && [ "$RUN_STEERED" = true ]; then
  echo "[INFO] === STEERED RUNS ==="

  BEST_LAYERS=("band:0.55:6" "band:0.15:6" "band:0.15:6" "band:0.75:6")
  BEST_STRENGTH=("1.5" "2.5" "2.0" "2.0")
  TS=$(date +"%Y%m%d-%H%M")

  for idx in 0 1 2 3; do
    if ! should_run_model_idx "$idx"; then
      continue
    fi

    MODEL_ID="${MODELS[$idx]}"
    MODEL_SLUG="${SLUGS[$idx]}"

    VEC_PATH="$(vector_path_for_model "$MODEL_SLUG" "python")"

    HP_LAYERS="${BEST_LAYERS[$idx]}"
    HP_STRENGTH="${BEST_STRENGTH[$idx]}"

    HP_LAYERS_TAG="L${HP_LAYERS//:/-}"
    HP_STRENGTH_TAG="a${HP_STRENGTH//./}"
    CONDITION="correctness_${HP_LAYERS_TAG}_${HP_STRENGTH_TAG}_${TS}"

    echo "[STEERED] $MODEL_ID ($CONDITION)"

    python gen.py \
      --model "$MODEL_ID" \
      --n "$N_SAMPLES" \
      --steer \
      --vector_path "$VEC_PATH" \
      --layers "$HP_LAYERS" \
      --strength "$HP_STRENGTH" \
      --dataset-tag "$BEHAVIOR_DATASET" \
      --condition "$CONDITION"
  done
fi

########################################
# EVALUATION
########################################

echo "[INFO] === EVALUATION ==="
python eval_all.py

echo "[INFO] DONE."

