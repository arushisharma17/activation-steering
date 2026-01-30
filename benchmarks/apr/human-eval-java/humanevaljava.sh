#!/usr/bin/env bash

set -euo pipefail

########################################
# Parse CLI arguments
########################################

RUN_BASELINE=false
RUN_STEERED=false

# Steering vectors source (kept as "tssb" only for anonymized release)
DATASET="tssb"

# Eval-only mode (skip generation, only run eval)
EVAL_ONLY=false

# pass@k control
MAX_K=""
K_LIST=""

# Model selection flags
RUN_QWEN_INST_7B=false
RUN_QWEN_CODER_7B=false
RUN_QWEN_CODER_14B=false
RUN_CODELLAMA_7B=false
EXPLICIT_MODELS=0   # 0 = no model flags yet, 1 = user specified at least one

# If no args passed, default = both (baseline + steered)
if [ $# -eq 0 ]; then
  RUN_BASELINE=true
  RUN_STEERED=true
fi

# Parse flags
while [[ $# -gt 0 ]]; do
  case $1 in
    --baseline)
      RUN_BASELINE=true; shift;;
    --steered)
      RUN_STEERED=true; shift;;

    # Eval-only mode
    --eval-only)
      EVAL_ONLY=true; shift;;

    # pass@k control
    --max-k)
      MAX_K="$2"; shift 2;;
    --k-list)
      K_LIST="$2"; shift 2;;

    # Model selection flags
    --qwen-inst-7b)
      RUN_QWEN_INST_7B=true; EXPLICIT_MODELS=1; shift;;
    --qwen-coder-7b)
      RUN_QWEN_CODER_7B=true; EXPLICIT_MODELS=1; shift;;
    --qwen-coder-14b)
      RUN_QWEN_CODER_14B=true; EXPLICIT_MODELS=1; shift;;
    --codellama-7b)
      RUN_CODELLAMA_7B=true; EXPLICIT_MODELS=1; shift;;

    *)
      echo "Unknown option: $1"
      exit 1;;
  esac
done

# If user didn’t specify any model flags, run all four
if [ "$EXPLICIT_MODELS" -eq 0 ]; then
  RUN_QWEN_INST_7B=true
  RUN_QWEN_CODER_7B=true
  RUN_QWEN_CODER_14B=true
  RUN_CODELLAMA_7B=true
fi

# If eval-only and no baseline/steered explicitly chosen, default to evaluating BOTH.
if [ "$EVAL_ONLY" = true ] && [ "$RUN_BASELINE" = false ] && [ "$RUN_STEERED" = false ]; then
  RUN_BASELINE=true
  RUN_STEERED=true
fi

# Build EVAL_KS list
EVAL_KS=()
if [ -n "$K_LIST" ]; then
  IFS=',' read -r -a EVAL_KS <<< "$K_LIST"
elif [ -n "$MAX_K" ]; then
  if ! [[ "$MAX_K" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] --max-k must be an integer (got '$MAX_K')."
    exit 1
  fi
  for ((k=1; k<=MAX_K; k++)); do EVAL_KS+=("$k"); done
else
  EVAL_KS=(1 5 10)
fi

echo "[INFO] CONFIG:"
echo "  DATASET            = $DATASET"
echo "  RUN_BASELINE       = $RUN_BASELINE"
echo "  RUN_STEERED        = $RUN_STEERED"
echo "  EVAL_ONLY          = $EVAL_ONLY"
echo "  EVAL_KS            = ${EVAL_KS[*]}"
echo "  RUN_QWEN_INST_7B   = $RUN_QWEN_INST_7B"
echo "  RUN_QWEN_CODER_7B  = $RUN_QWEN_CODER_7B"
echo "  RUN_QWEN_CODER_14B = $RUN_QWEN_CODER_14B"
echo "  RUN_CODELLAMA_7B   = $RUN_CODELLAMA_7B"
echo ""

########################################
# Repo-relative paths (no cluster/user info)
########################################

# Resolve repo root (works in anon artifact; fallback to current dir)
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
BENCH_DIR="${REPO_ROOT}/benchmarks/apr/human-eval-java"
cd "$BENCH_DIR"

mkdir -p logs
mkdir -p hej_generations

echo "[INFO] Repo root  : $REPO_ROOT"
echo "[INFO] Working dir: $(pwd)"

########################################
# Dependencies / environment notes
########################################
# This script assumes:
#  - Python environment can run meta/scripts/gen_hejava.py and eval scripts
#  - Java + Maven are available on PATH for HumanEval-Java compilation/tests
#
# If you use a virtualenv/conda, activate it before running this script.

########################################
# Helper: run eval for configured k values
########################################
run_eval_for_condition () {
  local MODEL_SLUG=$1
  local CONDITION=$2

  echo "[INFO] Evaluating condition '$CONDITION' for model '$MODEL_SLUG'..."
  for k in "${EVAL_KS[@]}"; do
    echo "[INFO]  -> pass@${k}"
    python meta/scripts/eval_hejava.py \
      --model_slug "$MODEL_SLUG" \
      --condition "$CONDITION" \
      --max_k "$k"
  done
}

########################################
# Model configuration (HF IDs + slugs)
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

# Vec IDs used only for naming steer-* dirs
VEC_IDS=(
  "tssb-qwen7b-general"
  "tssb-qwencoder7b"
  "tssb-qwencoder14b"
  "tssb-codellama7b"
)

########################################
# Steering vectors (repo-relative)
########################################
# NOTE: In anonymized artifacts, steering vectors are typically excluded from Git
# (large binaries). If you want steering enabled, place vectors locally under:
#   vectors/python/<...>.svec
#
# These are the expected filenames (edit if your local names differ).

VECTORS=(
  "${REPO_ROOT}/vectors/python/correctness_vector_100_qwen2-5-7b-instruct_pca_center_suffix.svec"
  "${REPO_ROOT}/vectors/python/correctness_vector_100_qwen2-5-coder-7b-instruct_pca_center_suffix.svec"
  "${REPO_ROOT}/vectors/python/correctness_vector_100_qwen2-5-coder-14b-instruct_pca_center_suffix.svec"
  "${REPO_ROOT}/vectors/python/correctness_vector_100_codellama-7b-instruct-hf_pca_center_suffix.svec"
)

N_MODELS=${#MODELS[@]}

########################################
# Best hyperparameters per model (from MCQ tuning on TSSB)
########################################

BEST_LAYERS=(
  "band:0.55:6"
  "band:0.15:6"
  "band:0.15:6"
  "band:0.75:6"
)

BEST_STRENGTHS=(
  1.5
  2.5
  2.0
  2.0
)

echo "[INFO] Using hyperparameters per model:"
for ((i=0; i<"$N_MODELS"; i++)); do
  echo "  ${SLUGS[$i]}: layers=${BEST_LAYERS[$i]}, strength=${BEST_STRENGTHS[$i]}"
done

########################################
# 1) BASELINE RUNS (no steering)
########################################

if [ "$RUN_BASELINE" = true ]; then
  echo "[INFO] === Baseline (no steering) generation + eval for selected models ==="

  for ((i=0; i<"$N_MODELS"; i++)); do
    # Skip models the user did not select
    if   [ "$i" -eq 0 ] && [ "$RUN_QWEN_INST_7B" != true ]; then continue
    elif [ "$i" -eq 1 ] && [ "$RUN_QWEN_CODER_7B" != true ]; then continue
    elif [ "$i" -eq 2 ] && [ "$RUN_QWEN_CODER_14B" != true ]; then continue
    elif [ "$i" -eq 3 ] && [ "$RUN_CODELLAMA_7B" != true ]; then continue
    fi

    MODEL_ID="${MODELS[$i]}"
    MODEL_SLUG="${SLUGS[$i]}"

    echo
    echo "[INFO] --- BASELINE: $MODEL_ID ($MODEL_SLUG) ---"

    if [ "$EVAL_ONLY" = false ]; then
      echo "[INFO] Generating baseline completions (n=10)..."
      python meta/scripts/gen_hejava.py \
        --model "$MODEL_ID" \
        --n 10
      echo "[INFO] Baseline generation done for $MODEL_SLUG."
    else
      echo "[INFO] EVAL_ONLY=true → skipping baseline generation for $MODEL_SLUG."
    fi

    echo "[INFO] Running baseline evaluation..."
    run_eval_for_condition "$MODEL_SLUG" "baseline"
  done

  echo "[INFO] All baseline runs finished."
fi

########################################
# 2) STEERED RUNS
########################################

if [ "$RUN_STEERED" = true ]; then
  echo "[INFO] === Steered generation + eval for selected models ==="

  # Basic existence check for vectors (fail early with a helpful message)
  for p in "${VECTORS[@]}"; do
    if [[ ! -f "$p" ]]; then
      echo "[ERROR] Missing steering vector: $p"
      echo "        Place vectors locally under ./vectors/python/ (not tracked in Git) or disable --steered."
      exit 1
    fi
  done

  for ((i=0; i<"$N_MODELS"; i++)); do
    # Skip models the user did not select
    if   [ "$i" -eq 0 ] && [ "$RUN_QWEN_INST_7B" != true ]; then continue
    elif [ "$i" -eq 1 ] && [ "$RUN_QWEN_CODER_7B" != true ]; then continue
    elif [ "$i" -eq 2 ] && [ "$RUN_QWEN_CODER_14B" != true ]; then continue
    elif [ "$i" -eq 3 ] && [ "$RUN_CODELLAMA_7B" != true ]; then continue
    fi

    MODEL_ID="${MODELS[$i]}"
    MODEL_SLUG="${SLUGS[$i]}"
    VEC_PATH="${VECTORS[$i]}"
    VEC_ID="${VEC_IDS[$i]}"
    LAYERS="${BEST_LAYERS[$i]}"
    STRENGTH="${BEST_STRENGTHS[$i]}"

    echo
    echo "[INFO] --- STEERED: $MODEL_ID ($MODEL_SLUG) ---"
    echo "[INFO] Using steering vector: $VEC_PATH (vec_id=$VEC_ID)"
    echo "[INFO] Using hyperparameters: layers=${LAYERS}, strength=${STRENGTH}"

    if [ "$EVAL_ONLY" = false ]; then
      python meta/scripts/gen_hejava.py \
        --model "$MODEL_ID" \
        --n 10 \
        --steer \
        --vector_path "$VEC_PATH" \
        --strength "$STRENGTH" \
        --layers "$LAYERS" \
        --vec_id "$VEC_ID"

      echo "[INFO] Steered generation done for $MODEL_SLUG."
    else
      echo "[INFO] EVAL_ONLY=true → skipping steered generation for $MODEL_SLUG."
    fi

    # Auto-detect newest steer-* directory for this model
    STEER_COND=$(ls -td "hej_generations/$MODEL_SLUG"/steer-* 2>/dev/null | head -n 1 | xargs -n 1 basename 2>/dev/null || echo "")
    if [ -z "$STEER_COND" ]; then
      echo "[WARN] No steered condition directory found for $MODEL_SLUG; skipping steered eval."
      continue
    fi

    echo "[INFO] Auto-detected steered condition for $MODEL_SLUG: $STEER_COND"
    run_eval_for_condition "$MODEL_SLUG" "$STEER_COND"
  done

  echo "[INFO] All steered runs finished."
fi

echo
echo "[INFO] Running HEJava summary aggregation..."
python meta/scripts/summarize_hejava_results.py || echo "[WARN] Summary aggregation failed."

