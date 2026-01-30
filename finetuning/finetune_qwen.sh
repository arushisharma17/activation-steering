#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}/finetuning"

mkdir -p logs runs

if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  source "${REPO_ROOT}/.venv/bin/activate"
fi

DATA="${REPO_ROOT}/data/tssb_data_3M/filtered-1.jsonl"
MODELS="qwen7b,coder7b,coder14b"

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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODELS="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --max_len) MAX_LEN="$2"; shift 2 ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      exit 1
      ;;
  esac
done

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

declare -A MODEL_MAP
MODEL_MAP[qwen7b]="Qwen/Qwen2.5-7B-Instruct"
MODEL_MAP[coder7b]="Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_MAP[coder14b]="Qwen/Qwen2.5-Coder-14B-Instruct"

IFS=',' read -ra MODEL_KEYS <<< "${MODELS}"

echo "[INFO] Models:  ${MODEL_KEYS[*]}"
echo "[INFO] Data:    ${DATA}"

for KEY in "${MODEL_KEYS[@]}"; do
  if [[ -z "${MODEL_MAP[$KEY]:-}" ]]; then
    echo "[ERROR] Unknown model key: ${KEY} (valid: qwen7b, coder7b, coder14b)"
    exit 1
  fi

  MODEL="${MODEL_MAP[$KEY]}"
  OUTDIR="./runs/${KEY}_qlora"
  mkdir -p "${OUTDIR}"

  echo "============================================================"
  echo "[INFO] Model key: ${KEY}"
  echo "[INFO] Model:     ${MODEL}"
  echo "[INFO] Outdir:    ${OUTDIR}"
  echo "============================================================"

  python qwen_finetune.py \
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

  echo "[DONE] Finished ${KEY}"
done

echo "[DONE] Completed."

