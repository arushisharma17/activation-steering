#!/bin/bash

#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:a100
#SBATCH --mem=128G
#SBATCH --job-name="mcq_window_sweep"
#SBATCH --mail-user=arushi17@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="logs/slurm-mcq-window-%j.out"

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

export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

cd activation-steering/

########################################
# Defaults / CLI args
########################################

DATASET="tssb"          # tssb | manysstubs
FEWSHOT_K=1
SEED=42
EVAL_START=0            # index of first eval item
EVAL_LIMIT=5000            # 0 = all items from EVAL_START

# Models to run (keys: codellama-7b qwen-inst-7b qwen-coder-7b qwen-coder-14b)
MODELS_STR="codellama-7b qwen-coder-7b qwen-coder-14b qwen-inst-7b"

# Strengths to sweep
STRENGTHS=(1.0 1.5 2.0 2.5 3.0)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --fewshot-k|--k)
      FEWSHOT_K="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --eval-start)
      EVAL_START="$2"
      shift 2
      ;;
    --eval-limit)
      EVAL_LIMIT="$2"
      shift 2
      ;;
    --models)
      # e.g. "codellama-7b qwen-coder-7b"
      MODELS_STR="$2"
      shift 2
      ;;
    --strengths)
      IFS=' ' read -r -a STRENGTHS <<< "$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

########################################
# Folder structure + MCQ file
########################################

#ROOT="mcq_cache/${DATASET}"

#mkdir -p "${ROOT}/mcq" "${ROOT}/baseline" "${ROOT}/steered"

#MCQ_FILE="${ROOT}/mcq/${DATASET}_mcq_k${FEWSHOT_K}_seed${SEED}.json"


########################################
# Folder structure + MCQ file (steering_100)
########################################

# Root for this dataset's MCQ cache under the 100-sample steering tree
ROOT="mcq_cache/steering_100/${DATASET}"

# Tell ab_apr_eval.py to write caches + metrics here
export MCQ_CACHE_DIR="${ROOT}"

mkdir -p "${ROOT}/mcq" "${ROOT}/baseline" "${ROOT}/steered"

MCQ_FILE="${ROOT}/mcq/${DATASET}_mcq_k${FEWSHOT_K}_seed${SEED}.json"


if [[ ! -f "${MCQ_FILE}" ]]; then
  echo "[ERROR] MCQ file not found: ${MCQ_FILE}"
  echo "        Build it first (e.g., with your MCQ builder)."
  exit 1
fi

########################################
# Model slugs (for consistency elsewhere, if needed)
########################################

CODELLAMA_SLUG="codellama-7b-instruct-hf"
QWEN_INST_7B_SLUG="qwen2-5-7b-instruct"
QWEN_CODER_7B_SLUG="qwen2-5-coder-7b-instruct"
QWEN_CODER_14B_SLUG="qwen2-5-coder-14b-instruct"

########################################
# Narrow layer windows per model
# NOTE: parser expects comma-separated 0-based indices.
########################################

# CodeLlama-7B: around 19–23
CODELLAMA_WINDOWS=(
  "17,18,19,20,21"
  "19,20,21,22,23"
  "21,22,23,24,25"
)

# Qwen2.5-Coder-7B: around 8–12
QWEN_CODER_7B_WINDOWS=(
  "6,7,8,9,10"
  "8,9,10,11,12"
  "10,11,12,13,14"
)

# Qwen2.5-Coder-14B: around 21–25
QWEN_CODER_14B_WINDOWS=(
  "19,20,21,22,23"
  "21,22,23,24,25"
  "23,24,25,26,27"
)

# Qwen2.5-7B-Instruct (if you want it)
QWEN_INST_7B_WINDOWS=(
  "14,15,16,17,18"
  "16,17,18,19,20"
  "18,19,20,21,22"
)

########################################
# Helper: run sweep for a single model
########################################

run_windows_for_model () {
  local model_label="$1"    # e.g. "codellama-7b"
  local model_flag="$2"     # e.g. "--codellama-7b"
  local -n windows_ref="$3" # nameref to WINDOWS array

  echo "--------------------------------------------"
  echo "[MODEL] ${model_label}"
  echo "--------------------------------------------"

  for LAY in "${windows_ref[@]}"; do
    for A in "${STRENGTHS[@]}"; do
      echo ">>> ${model_label} | layers=${LAY} | strength=${A}"
      bash slurm/run_mcq_eval.sh \
        --dataset "${DATASET}" \
        --mode steered \
        --layers "${LAY}" \
        --strength "${A}" \
        --fewshot-k "${FEWSHOT_K}" \
        --seed "${SEED}" \
        --mcq-file "${MCQ_FILE}" \
        --eval-start "${EVAL_START}" \
        --eval-limit "${EVAL_LIMIT}" \
        ${model_flag}
    done
  done
}

########################################
# Decide which models to run
########################################

RUN_CODELLAMA=0
RUN_QWEN_INST_7B=0
RUN_QWEN_CODER_7B=0
RUN_QWEN_CODER_14B=0

for m in $MODELS_STR; do
  case "$m" in
    codellama-7b|codellama)
      RUN_CODELLAMA=1
      ;;
    qwen-inst-7b)
      RUN_QWEN_INST_7B=1
      ;;
    qwen-coder-7b)
      RUN_QWEN_CODER_7B=1
      ;;
    qwen-coder-14b)
      RUN_QWEN_CODER_14B=1
      ;;
    *)
      echo "[WARN] Unknown model key in --models: $m"
      ;;
  esac
done

echo "MCQ narrow-window sweep"
echo "  Dataset    : ${DATASET}"
echo "  MCQ file   : ${MCQ_FILE}"
echo "  Few-shot k : ${FEWSHOT_K}"
echo "  Seed       : ${SEED}"
echo "  Eval start : ${EVAL_START}"
echo "  Eval limit : ${EVAL_LIMIT}"
echo "  Models     : ${MODELS_STR}"
echo "  Strengths  : ${STRENGTHS[@]}"
echo "============================================"

########################################
# Run selected models
########################################

if [[ "$RUN_CODELLAMA" -eq 1 ]]; then
  run_windows_for_model "codellama-7b" "--codellama-7b" CODELLAMA_WINDOWS
fi

if [[ "$RUN_QWEN_INST_7B" -eq 1 ]]; then
  run_windows_for_model "qwen-inst-7b" "--qwen-inst-7b" QWEN_INST_7B_WINDOWS
fi

if [[ "$RUN_QWEN_CODER_7B" -eq 1 ]]; then
  run_windows_for_model "qwen-coder-7b" "--qwen-coder-7b" QWEN_CODER_7B_WINDOWS
fi

if [[ "$RUN_QWEN_CODER_14B" -eq 1 ]]; then
  run_windows_for_model "qwen-coder-14b" "--qwen-coder-14b" QWEN_CODER_14B_WINDOWS
fi

echo "[INFO] Narrow-window sweep finished."

