#!/usr/bin/env bash
set -euo pipefail

########################################
# Resolve repo root
########################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}/finetuning"

mkdir -p logs runs

########################################
# Optional virtualenv
########################################
if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  source "${REPO_ROOT}/.venv/bin/activate"
fi

########################################
# Dataset
########################################
DATA="${REPO_ROOT}/data/tssb_data_3M/filtered-1.jsonl"

########################################
# Hyperparameters
########################################
MODEL="codellama/CodeLlama-7b-Instruct-hf"
OUTDIR="./runs/codellama7b_qlora"

MAX_LEN=1024
EVAL_RATIO=0.10
SEED=42

EPOCHS=2
LR=1e-4
WARMUP=0.10

TRAIN_BS=2
EVAL_BS=2
GRAD_ACCUM=16

EVAL_STEPS=200
SAVE_STEPS=400
LOG_STEPS=50
EARLY_PATIENCE=3

LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGETS="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

USE_BF16=1
USE_FP16=0
GRAD_CKPT=1

########################################
# Precision flags
########################################
PREC_FLAGS=""
if [[ "${USE_BF16}" == "1" ]]; then
  PREC_FLAGS="--bf16"
elif [[ "${USE_FP16}" == "1" ]]; then
  PREC_FLAGS="--fp16"
fi

GC_FLAGS=""
if [[ "${GRAD_CKPT}" == "1" ]]; then
  GC_FLAGS="--gradient_checkpointing"
fi

########################################
# Run
########################################
echo "============================================================"
echo "[INFO] Model:   ${MODEL}"
echo "[INFO] Dataset: ${DATA}"
echo "[INFO] Outdir:  ${OUTDIR}"
echo "============================================================"

python codellama_finetune.py \
  --model "${MODEL}" \
  --data "${DATA}" \
  --output_dir "${OUTDIR}" \
  --max_length "${MAX_LEN}" \
  --eval_ratio "${EVAL_RATIO}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --warmup_ratio "${WARMUP}" \
  --train_bs "${TRAIN_BS}" \
  --eval_bs "${EVAL_BS}" \
  --grad_accum "${GRAD_ACCUM}" \
  --eval_steps "${EVAL_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --log_steps "${LOG_STEPS}" \
  --early_stopping_patience "${EARLY_PATIENCE}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --lora_targets "${LORA_TARGETS}" \
  ${PREC_FLAGS} \
  ${GC_FLAGS} \
  --sanity_n 1

echo "[DONE] Fine-tuning complete."

