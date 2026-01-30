#!/usr/bin/env bash
#
# Portable summarization runner (baseline / steered / eval)
# Assumes Python environment already has required deps installed.
#
# Usage:
#   ./run_summarization_all.sh --baseline
#   ./run_summarization_all.sh --steered
#   ./run_summarization_all.sh --eval
#   ./run_summarization_all.sh --baseline --steered --eval
#
# Optional:
#   export PROJECT_ROOT=/path/to/repo
#   export STEERING_ROOT=/path/to/steering/library   (if needed by utils.py)
#

set -euo pipefail

########################################
# Parse CLI flags
########################################
RUN_BASELINE=0
RUN_STEERED=0
RUN_EVAL=0
MAX_EXAMPLES=5000

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --baseline) RUN_BASELINE=1 ;;
    --steered)  RUN_STEERED=1 ;;
    --eval)     RUN_EVAL=1 ;;
    --max_examples) MAX_EXAMPLES="$2"; shift ;;
    *)
      echo "[ERROR] Unknown flag: $1"
      exit 1
      ;;
  esac
  shift
done

if [[ "$RUN_BASELINE" -eq 0 && "$RUN_STEERED" -eq 0 && "$RUN_EVAL" -eq 0 ]]; then
  echo "[ERROR] No action specified. Use one or more of: --baseline --steered --eval"
  exit 1
fi

echo "[INFO] FLAGS:"
echo "  baseline     = $RUN_BASELINE"
echo "  steered      = $RUN_STEERED"
echo "  eval         = $RUN_EVAL"
echo "  max_examples = $MAX_EXAMPLES"
echo ""

########################################
# Project paths (system-agnostic)
########################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

SUM_ROOT="${PROJECT_ROOT}/benchmarks/generalization/summarization"
VEC_ROOT="${PROJECT_ROOT}/vectors"

if [[ ! -d "$SUM_ROOT" ]]; then
  echo "[ERROR] Summarization directory not found: $SUM_ROOT"
  echo "Set PROJECT_ROOT to your repo root (export PROJECT_ROOT=/path/to/repo)."
  exit 1
fi

cd "$SUM_ROOT"
mkdir -p results logs

########################################
# Helper: placeholder vector locations
########################################
# Expected layout (placeholders):
#   $VEC_ROOT/python/Qwen2.5-7B-Instruct.svec
#   $VEC_ROOT/python/Qwen2.5-Coder-7B-Instruct.svec
#   ...
#
vector_path_for_model() {
  local model_slug="$1"
  local lang="$2"   # python | java
  echo "${VEC_ROOT}/${lang}/${model_slug}.svec"
}

########################################
# Helper: auto-naming steered outputs
########################################
steered_filename () {
  local tag="$1"
  local layers="$2"
  local strength="$3"

  local LAYER_SAFE="${layers//:/_}"
  local STRENGTH_SAFE="${strength//./_}"

  echo "results/codexglue_python_${tag}_steered_l${LAYER_SAFE}_s${STRENGTH_SAFE}_${MAX_EXAMPLES}.jsonl"
}

run_steered_model () {
  local model_name="$1"
  local model_slug="$2"
  local tag="$3"
  local layers="$4"
  local strength="$5"

  local vector_path
  vector_path="$(vector_path_for_model "$model_slug" "python")"

  local outfile
  outfile="$(steered_filename "$tag" "$layers" "$strength")"

  echo "[INFO] Running steered summarization: ${tag}"
  echo "[INFO]   Model    = ${model_name}"
  echo "[INFO]   Vector   = ${vector_path}"
  echo "[INFO]   Layers   = ${layers}"
  echo "[INFO]   Strength = ${strength}"
  echo "[INFO]   Limit    = ${MAX_EXAMPLES}"
  echo "[INFO]   Output   = ${outfile}"

  python summarize_codexglue.py \
    --model "${model_name}" \
    --lang python \
    --split test \
    --limit "${MAX_EXAMPLES}" \
    --out "${outfile}" \
    --steer \
    --vector_path "${vector_path}" \
    --layers "${layers}" \
    --strength "${strength}"
}

########################################
# Models + steering hyperparameters
# (values are treated as fixed configuration for this script)
########################################

MODEL_QWEN7B="Qwen/Qwen2.5-7B-Instruct"
SLUG_QWEN7B="Qwen2.5-7B-Instruct"
TAG_QWEN7B="qwen7b"

MODEL_QWEN_CODER7B="Qwen/Qwen2.5-Coder-7B-Instruct"
SLUG_QWEN_CODER7B="Qwen2.5-Coder-7B-Instruct"
TAG_QWEN_CODER7B="qwen_coder7b"

MODEL_QWEN_CODER14B="Qwen/Qwen2.5-Coder-14B-Instruct"
SLUG_QWEN_CODER14B="Qwen2.5-Coder-14B-Instruct"
TAG_QWEN_CODER14B="qwen_coder14b"

MODEL_CODELLAMA7B="meta-llama/CodeLlama-7b-Instruct-hf"
SLUG_CODELLAMA7B="CodeLlama-7b-Instruct-hf"
TAG_CODELLAMA7B="codellama7b"

# Fixed layer/strength settings
LAYERS_QWEN7B="band:0.55:6"
STRENGTH_QWEN7B="1.5"

LAYERS_QWEN_CODER7B="band:0.15:6"
STRENGTH_QWEN_CODER7B="2.5"

LAYERS_QWEN_CODER14B="band:0.15:6"
STRENGTH_QWEN_CODER14B="2.0"

LAYERS_CODELLAMA7B="band:0.75:6"
STRENGTH_CODELLAMA7B="2.0"

########################################
# BASELINE RUNS
########################################
if [[ "$RUN_BASELINE" -eq 1 ]]; then
  echo "[INFO] Running baseline summarization (limit=${MAX_EXAMPLES})..."

  python summarize_codexglue.py \
    --model "${MODEL_QWEN_CODER7B}" \
    --lang python \
    --split test \
    --limit "${MAX_EXAMPLES}" \
    --out "results/codexglue_python_${TAG_QWEN_CODER7B}_baseline_${MAX_EXAMPLES}.jsonl"

  python summarize_codexglue.py \
    --model "${MODEL_QWEN7B}" \
    --lang python \
    --split test \
    --limit "${MAX_EXAMPLES}" \
    --out "results/codexglue_python_${TAG_QWEN7B}_baseline_${MAX_EXAMPLES}.jsonl"

  python summarize_codexglue.py \
    --model "${MODEL_QWEN_CODER14B}" \
    --lang python \
    --split test \
    --limit "${MAX_EXAMPLES}" \
    --out "results/codexglue_python_${TAG_QWEN_CODER14B}_baseline_${MAX_EXAMPLES}.jsonl"

  python summarize_codexglue.py \
    --model "${MODEL_CODELLAMA7B}" \
    --lang python \
    --split test \
    --limit "${MAX_EXAMPLES}" \
    --out "results/codexglue_python_${TAG_CODELLAMA7B}_baseline_${MAX_EXAMPLES}.jsonl"
fi

########################################
# STEERED RUNS
########################################
if [[ "$RUN_STEERED" -eq 1 ]]; then
  echo "[INFO] Running steered summarization on ${MAX_EXAMPLES} examples..."

  run_steered_model \
    "${MODEL_QWEN_CODER7B}" \
    "${SLUG_QWEN_CODER7B}" \
    "${TAG_QWEN_CODER7B}" \
    "${LAYERS_QWEN_CODER7B}" \
    "${STRENGTH_QWEN_CODER7B}"

  run_steered_model \
    "${MODEL_QWEN7B}" \
    "${SLUG_QWEN7B}" \
    "${TAG_QWEN7B}" \
    "${LAYERS_QWEN7B}" \
    "${STRENGTH_QWEN7B}"

  run_steered_model \
    "${MODEL_QWEN_CODER14B}" \
    "${SLUG_QWEN_CODER14B}" \
    "${TAG_QWEN_CODER14B}" \
    "${LAYERS_QWEN_CODER14B}" \
    "${STRENGTH_QWEN_CODER14B}"

  run_steered_model \
    "${MODEL_CODELLAMA7B}" \
    "${SLUG_CODELLAMA7B}" \
    "${TAG_CODELLAMA7B}" \
    "${LAYERS_CODELLAMA7B}" \
    "${STRENGTH_CODELLAMA7B}"
fi

########################################
# EVALUATION
########################################
if [[ "$RUN_EVAL" -eq 1 ]]; then
  echo "[INFO] Starting evaluation on ${MAX_EXAMPLES} examples..."

  # Baseline
  python eval_summarization.py --pred_file "results/codexglue_python_${TAG_QWEN_CODER7B}_baseline_${MAX_EXAMPLES}.jsonl"  --lang python
  python eval_summarization.py --pred_file "results/codexglue_python_${TAG_QWEN7B}_baseline_${MAX_EXAMPLES}.jsonl"        --lang python
  python eval_summarization.py --pred_file "results/codexglue_python_${TAG_QWEN_CODER14B}_baseline_${MAX_EXAMPLES}.jsonl" --lang python
  python eval_summarization.py --pred_file "results/codexglue_python_${TAG_CODELLAMA7B}_baseline_${MAX_EXAMPLES}.jsonl"   --lang python

  # Steered (filenames encode hyperparameters)
  python eval_summarization.py --pred_file "$(steered_filename "${TAG_QWEN_CODER7B}"   "${LAYERS_QWEN_CODER7B}"   "${STRENGTH_QWEN_CODER7B}")"   --lang python
  python eval_summarization.py --pred_file "$(steered_filename "${TAG_QWEN7B}"         "${LAYERS_QWEN7B}"         "${STRENGTH_QWEN7B}")"         --lang python
  python eval_summarization.py --pred_file "$(steered_filename "${TAG_QWEN_CODER1_

