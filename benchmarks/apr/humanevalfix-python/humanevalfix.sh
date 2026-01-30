#!/bin/bash
#
# Script: run_hefix_all_models.sh (ANON / PORTABLE)
#
# HumanEvalFix-Python multi-model runner
# Baseline + steered (TSSB-only)
#

set -euo pipefail

########################################
# Locate repo root + move to benchmark dir
########################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# This script lives in: benchmarks/apr/humanevalfix-python/
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p logs
mkdir -p hefix_generations

########################################
# Optional: env activation (generic)
########################################
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
# Defaults
########################################
RUN_BASELINE=false
RUN_STEERED=false
DATASET="tssb"          # only supported option in anonymized artifact
EVAL_ONLY=false

# Generation / eval knobs
N_SAMPLES=10
MAX_NEW_TOKENS=1024
TEMPERATURE=0.2
TOP_P=0.95

MAX_TRIES=3
MIN_CHARS=40
RETRY_TEMP_MULT=1.25

NO_OVERWRITE=false

# eval pass@k list
MAX_K=""
K_LIST=""

# model selection flags
RUN_QWEN_INST_7B=false
RUN_QWEN_CODER_7B=false
RUN_QWEN_CODER_14B=false
RUN_CODELLAMA_7B=false
EXPLICIT_MODELS=0

# If no args: run both baseline and steered
if [[ $# -eq 0 ]]; then
  RUN_BASELINE=true
  RUN_STEERED=true
fi

########################################
# Parse CLI arguments
########################################
while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline) RUN_BASELINE=true; shift ;;
    --steered)  RUN_STEERED=true; shift ;;
    --dataset)  DATASET="$2"; shift 2 ;;
    --eval-only) EVAL_ONLY=true; shift ;;

    --n) N_SAMPLES="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --top-p) TOP_P="$2"; shift 2 ;;
    --max-tries) MAX_TRIES="$2"; shift 2 ;;
    --min-chars) MIN_CHARS="$2"; shift 2 ;;
    --retry-temp-mult) RETRY_TEMP_MULT="$2"; shift 2 ;;
    --no-overwrite) NO_OVERWRITE=true; shift ;;

    --max-k) MAX_K="$2"; shift 2 ;;
    --k-list) K_LIST="$2"; shift 2 ;;

    --qwen-inst-7b) RUN_QWEN_INST_7B=true; EXPLICIT_MODELS=1; shift ;;
    --qwen-coder-7b) RUN_QWEN_CODER_7B=true; EXPLICIT_MODELS=1; shift ;;
    --qwen-coder-14b) RUN_QWEN_CODER_14B=true; EXPLICIT_MODELS=1; shift ;;
    --codellama-7b) RUN_CODELLAMA_7B=true; EXPLICIT_MODELS=1; shift ;;

    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

########################################
# Validation
########################################
if [[ "${DATASET}" != "tssb" ]]; then
  echo "[ERROR] This anonymous artifact supports only --dataset tssb."
  exit 1
fi

if [[ "${EXPLICIT_MODELS}" -eq 0 ]]; then
  RUN_QWEN_INST_7B=true
  RUN_QWEN_CODER_7B=true
  RUN_QWEN_CODER_14B=true
  RUN_CODELLAMA_7B=true
fi

if [[ "${EVAL_ONLY}" == true && "${RUN_BASELINE}" == false && "${RUN_STEERED}" == false ]]; then
  RUN_BASELINE=true
  RUN_STEERED=true
fi

########################################
# Build eval k list
########################################
EVAL_KS=()
if [[ -n "${K_LIST}" ]]; then
  IFS=',' read -r -a EVAL_KS <<< "${K_LIST}"
elif [[ -n "${MAX_K}" ]]; then
  if ! [[ "${MAX_K}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] --max-k must be an integer (got '${MAX_K}')."
    exit 1
  fi
  for ((k=1; k<=MAX_K; k++)); do EVAL_KS+=("${k}"); done
else
  EVAL_KS=(1 5 10)
fi
KCSV=$(IFS=, ; echo "${EVAL_KS[*]}")

########################################
# Model config
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

VEC_ROOT="${ROOT_DIR}/vectors"

echo "[INFO] Using TSSB-trained steering vectors."
VECTORS=(
  "${VEC_ROOT}/python/correctness_vector_100_qwen2-5-7b-instruct_pca_center_suffix.svec"
  "${VEC_ROOT}/python/correctness_vector_100_qwen2-5-coder-7b-instruct_pca_center_suffix.svec"
  "${VEC_ROOT}/python/correctness_vector_100_qwen2-5-coder-14b-instruct_pca_center_suffix.svec"
  "${VEC_ROOT}/python/correctness_vector_100_codellama-7b-instruct-hf_pca_center_suffix.svec"
)

# Keep IDs generic (no dataset tags that could encode lab naming conventions)
VEC_IDS=("vec0" "vec1" "vec2" "vec3")

N_MODELS=${#MODELS[@]}

# Hyperparameters used in the experiments (from your script)
BEST_LAYERS=("band:0.55:6" "band:0.15:6" "band:0.15:6" "band:0.75:6")
BEST_STRENGTHS=(1.5 2.5 2.0 2.0)

########################################
# Helpers
########################################
run_eval () {
  local MODEL_SLUG="$1"
  local CONDITION="$2"
  python eval_hefix.py --model_slug "${MODEL_SLUG}" --condition "${CONDITION}" --k_list "${KCSV}"
}

GEN_FLAGS=(
  --n "${N_SAMPLES}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --max_tries "${MAX_TRIES}"
  --min_chars "${MIN_CHARS}"
  --retry_temp_mult "${RETRY_TEMP_MULT}"
)

if [[ "${NO_OVERWRITE}" == true ]]; then
  GEN_FLAGS+=(--no_overwrite)
fi

echo "[INFO] CONFIG:"
echo "  DATASET            = ${DATASET}"
echo "  RUN_BASELINE       = ${RUN_BASELINE}"
echo "  RUN_STEERED        = ${RUN_STEERED}"
echo "  EVAL_ONLY          = ${EVAL_ONLY}"
echo "  N_SAMPLES          = ${N_SAMPLES}"
echo "  MAX_NEW_TOKENS     = ${MAX_NEW_TOKENS}"
echo "  TEMP / TOP_P       = ${TEMPERATURE} / ${TOP_P}"
echo "  MAX_TRIES          = ${MAX_TRIES}"
echo "  MIN_CHARS          = ${MIN_CHARS}"
echo "  RETRY_TEMP_MULT    = ${RETRY_TEMP_MULT}"
echo "  NO_OVERWRITE       = ${NO_OVERWRITE}"
echo "  EVAL_KS            = ${EVAL_KS[*]}"
echo "  RUN_QWEN_INST_7B   = ${RUN_QWEN_INST_7B}"
echo "  RUN_QWEN_CODER_7B  = ${RUN_QWEN_CODER_7B}"
echo "  RUN_QWEN_CODER_14B = ${RUN_QWEN_CODER_14B}"
echo "  RUN_CODELLAMA_7B   = ${RUN_CODELLAMA_7B}"
echo ""

########################################
# BASELINE
########################################
if [[ "${RUN_BASELINE}" == true ]]; then
  echo "[INFO] === BASELINE runs ==="
  for ((i=0; i<"${N_MODELS}"; i++)); do
    if   [[ "${i}" -eq 0 && "${RUN_QWEN_INST_7B}" != true ]]; then continue
    elif [[ "${i}" -eq 1 && "${RUN_QWEN_CODER_7B}" != true ]]; then continue
    elif [[ "${i}" -eq 2 && "${RUN_QWEN_CODER_14B}" != true ]]; then continue
    elif [[ "${i}" -eq 3 && "${RUN_CODELLAMA_7B}" != true ]]; then continue
    fi

    MODEL_ID="${MODELS[$i]}"
    MODEL_SLUG="${SLUGS[$i]}"

    echo
    echo "[INFO] --- BASELINE: ${MODEL_ID} (${MODEL_SLUG}) ---"

    if [[ "${EVAL_ONLY}" == false ]]; then
      python gen_hefix.py \
        --model "${MODEL_ID}" \
        --condition "baseline" \
        "${GEN_FLAGS[@]}"
    else
      echo "[INFO] EVAL_ONLY=true -> skipping baseline generation for ${MODEL_SLUG}"
    fi

    run_eval "${MODEL_SLUG}" "baseline"
  done
fi

########################################
# STEERED
########################################
if [[ "${RUN_STEERED}" == true ]]; then
  echo "[INFO] === STEERED runs (dataset=${DATASET}) ==="
  TS=$(date +"%Y%m%d-%H%M")

  for ((i=0; i<"${N_MODELS}"; i++)); do
    if   [[ "${i}" -eq 0 && "${RUN_QWEN_INST_7B}" != true ]]; then continue
    elif [[ "${i}" -eq 1 && "${RUN_QWEN_CODER_7B}" != true ]]; then continue
    elif [[ "${i}" -eq 2 && "${RUN_QWEN_CODER_14B}" != true ]]; then continue
    elif [[ "${i}" -eq 3 && "${RUN_CODELLAMA_7B}" != true ]]; then continue
    fi

    MODEL_ID="${MODELS[$i]}"
    MODEL_SLUG="${SLUGS[$i]}"
    VEC_PATH="${VECTORS[$i]}"
    VEC_ID="${VEC_IDS[$i]}"
    LAYERS="${BEST_LAYERS[$i]}"
    STRENGTH="${BEST_STRENGTHS[$i]}"

    if [[ ! -f "${VEC_PATH}" ]]; then
      echo "[ERROR] Missing vector file: ${VEC_PATH}"
      echo "        Ensure vectors are present under: ${VEC_ROOT}/python/"
      exit 1
    fi

    HP_LAYERS_TAG="L${LAYERS//:/-}"
    HP_STRENGTH_TAG="a${STRENGTH//./}"
    CONDITION="steer-${VEC_ID}_${HP_LAYERS_TAG}_${HP_STRENGTH_TAG}_${TS}"

    echo
    echo "[INFO] --- STEERED: ${MODEL_ID} (${MODEL_SLUG}) ---"
    echo "[INFO] layers=${LAYERS} strength=${STRENGTH}"
    echo "[INFO] condition=${CONDITION}"

    if [[ "${EVAL_ONLY}" == false ]]; then
      python gen_hefix.py \
        --model "${MODEL_ID}" \
        --condition "${CONDITION}" \
        --steer \
        --vector_path "${VEC_PATH}" \
        --layers "${LAYERS}" \
        --strength "${STRENGTH}" \
        "${GEN_FLAGS[@]}"
    else
      echo "[INFO] EVAL_ONLY=true -> skipping steered generation for ${MODEL_SLUG}"
    fi

    run_eval "${MODEL_SLUG}" "${CONDITION}"
  done
fi

echo
echo "[INFO] Aggregating summaries..."
python summarize_hefix_results.py || echo "[WARN] summarize_hefix_results.py failed"

# Deactivate if we activated
if declare -F deactivate >/dev/null 2>&1; then
  deactivate || true
fi

echo "[INFO] DONE."

