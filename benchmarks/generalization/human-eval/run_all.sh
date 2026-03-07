#!/bin/bash
#
# HumanEval multi-model runner with optional steering using APR behavior vectors
#

#SBATCH --time=3-23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --job-name="humaneval-all"
#SBATCH --mail-user=arushi17@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="logs/slurm-%j.out"

########################################
# GLOBAL — SET TOKEN FIRST
########################################

export HF_TOKEN=""

#Force HuggingFace everywhere
export HF_HUB_ENABLE_HF_TRANSFER="1"
export TRANSFORMERS_VERBOSITY="error"

########################################
# Parse CLI arguments
########################################

RUN_BASELINE=false
RUN_STEERED=false
EVAL_ONLY=false
BEHAVIOR_DATASET="tssb"
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
# Logging
########################################

echo "======================================================="
echo "[CONFIG]"
echo "RUN_BASELINE   = $RUN_BASELINE"
echo "RUN_STEERED    = $RUN_STEERED"
echo "EVAL_ONLY      = $EVAL_ONLY"
echo "BEHAVIOR_DATASET = $BEHAVIOR_DATASET"
echo "N_SAMPLES      = $N_SAMPLES"
echo "Models:"
echo "  Qwen2.5-7B            = $RUN_QWEN_INST_7B"
echo "  Qwen2.5-Coder-7B      = $RUN_QWEN_CODER_7B"
echo "  Qwen2.5-Coder-14B     = $RUN_QWEN_CODER_14B"
echo "  CodeLlama-7B-Instruct = $RUN_CODELLAMA_7B"
echo "======================================================="

########################################
# Environment
########################################

ROOT="/lustre/hdd/LAS/jannesar-lab/arushi/activation-steering"

export MPLCONFIGDIR="$ROOT/matplotlib"
export XDG_CACHE_HOME="$ROOT/cache"
export TRITON_CACHE_DIR="$ROOT/cache/triton"

HE_ROOT="${ROOT}/benchmarks/generalization/human-eval"
VEC_ROOT="${ROOT}/vectors"

# HuggingFace local cache target
export HF_HOME="${ROOT}/models"

cd "$HE_ROOT"
mkdir -p logs results_100

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

echo "[INFO] Node: $(hostname)"
echo "[INFO] HF_HOME = $HF_HOME"
nvidia-smi || echo "[WARN] No GPU visible"

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

VECTORS_TSSB=(
  "${VEC_ROOT}/python/correctness_vector_100_qwen2-5-7b-instruct_pca_center_suffix.svec"
  "${VEC_ROOT}/python/correctness_vector_100_qwen2-5-coder-7b-instruct_pca_center_suffix.svec"
  "${VEC_ROOT}/python/correctness_vector_100_qwen2-5-coder-14b-instruct_pca_center_suffix.svec"
  "${VEC_ROOT}/python/correctness_vector_100_codellama-7b-instruct-hf_pca_center_suffix.svec"
)

VECTORS_MANY=(
  "${VEC_ROOT}/java/behavior_manysstubs_apr_manysstubs_pairs_3k_qwen2-5-7b-instruct_pca_center_suffix.svec"
  "${VEC_ROOT}/java/behavior_manysstubs_apr_manysstubs_pairs_3k_qwen2-5-coder-7b-instruct_pca_center_suffix.svec"
  "${VEC_ROOT}/java/behavior_manysstubs_apr_manysstubs_pairs_3k_qwen2-5-coder-14b-instruct_pca_center_suffix.svec"
  "${VEC_ROOT}/java/behavior_manysstubs_apr_manysstubs_pairs_3k_codellama-7b-instruct-hf_pca_center_suffix.svec"
)

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
  source /lustre/hdd/LAS/jannesar-lab/arushi/.venv/bin/activate

  for idx in 0 1 2 3; do
    if ! should_run_model_idx "$idx"; then
      continue
    fi
    MODEL_ID="${MODELS[$idx]}"
    MODEL_SLUG="${SLUGS[$idx]}"
    echo "[BASELINE] $MODEL_ID"

    python gen.py \
      --model "$MODEL_ID" \
      --n "$N_SAMPLES" \
      --dataset-tag "none" \
      --condition "baseline"
  done

  deactivate
fi

########################################
# STEERED RUNS
########################################

if [ "$EVAL_ONLY" = false ] && [ "$RUN_STEERED" = true ]; then
  echo "[INFO] === STEERED RUNS ==="
  source /lustre/hdd/LAS/jannesar-lab/arushi/myenv/bin/activate

  BEST_LAYERS=("band:0.55:6" "band:0.15:6" "band:0.15:6" "band:0.75:6")
  BEST_STRENGTH=("1.5" "2.5" "2.0" "2.0")
  TS=$(date +"%Y%m%d-%H%M")

  for idx in 0 1 2 3; do
    if ! should_run_model_idx "$idx"; then
      continue
    fi

    MODEL_ID="${MODELS[$idx]}"
    MODEL_SLUG="${SLUGS[$idx]}"

    if [ "$BEHAVIOR_DATASET" = "tssb" ]; then
      VEC_PATH="${VECTORS_TSSB[$idx]}"
    else
      VEC_PATH="${VECTORS_MANY[$idx]}"
    fi

    HP_LAYERS="${BEST_LAYERS[$idx]}"
    HP_STRENGTH="${BEST_STRENGTH[$idx]}"

    HP_LAYERS_TAG="L${HP_LAYERS//:/-}"
    HP_STRENGTH_TAG="a${HP_STRENGTH//./}"
    CONDITION="correctness-${BEHAVIOR_DATASET}_${HP_LAYERS_TAG}_${HP_STRENGTH_TAG}_${TS}"

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

  deactivate
fi

########################################
# EVALUATION
########################################

echo "[INFO] === EVALUATION ==="
source /lustre/hdd/LAS/jannesar-lab/arushi/.venv/bin/activate
python eval_all.py
deactivate

echo "[INFO] DONE."

