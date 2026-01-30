#!/bin/bash
#SBATCH --time=3-05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --gres=gpu:a100
#SBATCH --mem=128G
#SBATCH --job-name="aprvec-merged-qwen"
#SBATCH --mail-user=arushi17@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
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

# Dataset (TSSB only)
TSSB_JSON="${ROOT}/data/tssb_data_3M/derived/correctness_behavior_apr.json"
DATASET_NAME="tssb"

# Training params
MAX_EXAMPLES=200
METHOD="pca_center"
LAST_TOKENS="suffix-only"

# Use merged finetuned models (local paths)
MERGED_ROOT="${ROOT}/finetuning/merged"

MODEL_QWEN_7B="${MERGED_ROOT}/qwen2.5-7b-instruct-bugfix-merged"
MODEL_CODER_7B="${MERGED_ROOT}/qwen2.5-coder-7b-instruct-bugfix-merged"
MODEL_CODER_14B="${MERGED_ROOT}/qwen2.5-coder-14b-instruct-bugfix-merged"

# If you want to keep HF cache separate for safety, set this explicitly:
HF_CACHE_DIR="${ROOT}/models"

########################################
# Helper
########################################
run_for_model() {
  local MODEL_PATH="$1"
  local TAG="$2"

  echo
  echo "======================================================="
  echo ">>> Training vector (MERGED FT): Dataset=${DATASET_NAME}, Model=${TAG}"
  echo "    JSON: ${TSSB_JSON}"
  echo "    Model path: ${MODEL_PATH}"
  echo "======================================================="

  ${PYTHON} "${TRAIN_SCRIPT}" \
    --pairs-json "${TSSB_JSON}" \
    --dataset-name "${DATASET_NAME}" \
    --max-examples "${MAX_EXAMPLES}" \
    --model-id "${MODEL_PATH}" \
    --tokenizer-id "${MODEL_PATH}" \
    --method "${METHOD}" \
    --last-tokens "${LAST_TOKENS}" \
    --hf-cache "${HF_CACHE_DIR}"
}

########################################
# Sanity checks
########################################
for d in "${MODEL_QWEN_7B}" "${MODEL_CODER_7B}" "${MODEL_CODER_14B}"; do
  if [[ ! -d "${d}" ]]; then
    echo "[ERROR] Missing merged model dir: ${d}"
    exit 1
  fi
done

########################################
# Run all 3 merged Qwen models
########################################
run_for_model "${MODEL_QWEN_7B}"   "Qwen2.5-7B-Instruct (bugfix merged)"
run_for_model "${MODEL_CODER_7B}"  "Qwen2.5-Coder-7B-Instruct (bugfix merged)"
run_for_model "${MODEL_CODER_14B}" "Qwen2.5-Coder-14B-Instruct (bugfix merged)"

echo "All merged-Qwen TSSB steering vectors completed."

