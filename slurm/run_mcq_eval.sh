#!/bin/bash

#SBATCH --time=5-05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --gres=gpu:a100
#SBATCH --mem=128G
#SBATCH --job-name="mcq_eval"
#SBATCH --mail-user=arushi17@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="logs/slurm-mcq-eval-%j.out"

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

nvidia-smi

export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

cd activation-steering/

########################################
# Parse script arguments
########################################
# Usage examples:
#   # Baseline only, all models, ManySStuBs
#   sbatch slurm/run_mcq_eval.sh \
#     --dataset manysstubs \
#     --mode baseline \
#     --fewshot-k 3 --seed 42
#
#   # Steered only, CodeLlama + Qwen coder 7B, TSSB vectors
#   sbatch slurm/run_mcq_eval.sh \
#     --dataset tssb \
#     --mode steered \
#     --fewshot-k 3 --seed 42 \
#     --layers "last:4" --strength 2.0 \
#     --codellama-7b --qwen-coder-7b
#
#   # Compare baseline vs steered (reuses cached baseline if present)
#   sbatch slurm/run_mcq_eval.sh \
#     --dataset manysstubs \
#     --mode compare \
#     --fewshot-k 3 --seed 42 \
#     --layers "last:4" --strength 2.0
#
#   # Small subset sanity check (first 100 eval items)
#   sbatch slurm/run_mcq_eval.sh \
#     --dataset manysstubs \
#     --mode compare \
#     --eval-start 0 --eval-limit 100
########################################

DATASET="manysstubs"   # tssb | manysstubs
MODE="compare"         # baseline | steered | compare
FEWSHOT_K=0
SEED=42
LAYERS="last:4"
STRENGTH="2.0"
MCQ_FILE=""            # if empty, use default mcq_cache/<dataset>_mcq_kK_seedS.json

# New: subset controls (pass-through to ab_apr_eval.py)
EVAL_START=0           # --eval_start
EVAL_LIMIT=0           # --eval_limit (0 = all remaining)

# model selection flags
RUN_CODELLAMA=0
RUN_QWEN_INST_7B=0
RUN_QWEN_CODER_7B=0
RUN_QWEN_CODER_14B=0
EXPLICIT_MODELS=0      # track whether user chose models

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"        # baseline | steered | compare
      shift 2
      ;;
    --fewshot-k)
      FEWSHOT_K="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --layers)
      LAYERS="$2"
      shift 2
      ;;
    --strength)
      STRENGTH="$2"
      shift 2
      ;;
    --mcq-file)
      MCQ_FILE="$2"
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
    # model selection
    --codellama-7b)
      RUN_CODELLAMA=1
      EXPLICIT_MODELS=1
      shift
      ;;
    --qwen-inst-7b)
      RUN_QWEN_INST_7B=1
      EXPLICIT_MODELS=1
      shift
      ;;
    --qwen-coder-7b)
      RUN_QWEN_CODER_7B=1
      EXPLICIT_MODELS=1
      shift
      ;;
    --qwen-coder-14b)
      RUN_QWEN_CODER_14B=1
      EXPLICIT_MODELS=1
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Default MCQ file if not specified
if [[ -z "$MCQ_FILE" ]]; then
  MCQ_FILE="mcq_cache/${DATASET}_mcq_k${FEWSHOT_K}_seed${SEED}.json"
fi

if [[ ! -f "$MCQ_FILE" ]]; then
  echo "ERROR: MCQ file not found: ${MCQ_FILE}"
  echo "       Build it first with slurm/run_mcq_build.sh (or ab_apr_eval --save_json)."
  exit 1
fi

# If user didn't explicitly pick models, run all four
if [[ "$EXPLICIT_MODELS" -eq 0 ]]; then
  RUN_CODELLAMA=1
  RUN_QWEN_INST_7B=1
  RUN_QWEN_CODER_7B=1
  RUN_QWEN_CODER_14B=1
fi

########################################
# Dataset-specific vector naming
########################################

case "$DATASET" in
  tssb)
    DATASET_SLUG="tssb"
    VECTOR_ROOT="vectors/python"
    VECTOR_PREFIX="correctness_vector_100"
    #VECTOR_PREFIX="behavior_tssb_correctness_behavior_apr"
    ;;
  manysstubs)
    DATASET_SLUG="manysstubs"
    VECTOR_ROOT="vectors/java"
    VECTOR_PREFIX="behavior_manysstubs_apr_manysstubs_pairs_3k"
    ;;
  *)
    echo "ERROR: --dataset must be 'tssb' or 'manysstubs' (got '$DATASET')."
    exit 1
    ;;
esac

VECTOR_SUFFIX="pca_center_suffix"

########################################
# Model IDs and slugs (match vector naming)
########################################

# 1) CodeLlama-7B-Instruct
CODELLAMA_MODEL="meta-llama/CodeLlama-7b-Instruct-hf"
CODELLAMA_TOKENIZER="$CODELLAMA_MODEL"
CODELLAMA_SLUG="codellama-7b-instruct-hf"

# 2) Qwen2.5-7B-Instruct
QWEN_INST_7B_MODEL="Qwen/Qwen2.5-7B-Instruct"
QWEN_INST_7B_TOKENIZER="$QWEN_INST_7B_MODEL"
QWEN_INST_7B_SLUG="qwen2-5-7b-instruct"

# 3) Qwen2.5-Coder-7B-Instruct
QWEN_CODER_7B_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
QWEN_CODER_7B_TOKENIZER="$QWEN_CODER_7B_MODEL"
QWEN_CODER_7B_SLUG="qwen2-5-coder-7b-instruct"

# 4) Qwen2.5-Coder-14B-Instruct
QWEN_CODER_14B_MODEL="Qwen/Qwen2.5-Coder-14B-Instruct"
QWEN_CODER_14B_TOKENIZER="$QWEN_CODER_14B_MODEL"
QWEN_CODER_14B_SLUG="qwen2-5-coder-14b-instruct"

########################################
# Shared eval settings
########################################

SHOW_N=6

# MODE -> ab_apr_eval flags
MODE_ARGS=()
case "$MODE" in
  compare)
    MODE_ARGS+=(--compare)
    ;;
  baseline)
    MODE_ARGS+=(--mode baseline)
    ;;
  steered)
    MODE_ARGS+=(--mode steered)
    ;;
  *)
    echo "ERROR: --mode must be one of: compare | baseline | steered"
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

  # Pick the right vector name pattern based on DATASET
  local vec_path
  if [[ "$DATASET" == "tssb" ]]; then
    vec_path="${VECTOR_ROOT}/${VECTOR_PREFIX}_${model_slug}_${VECTOR_SUFFIX}.svec"
  else
    vec_path="${VECTOR_ROOT}/${VECTOR_PREFIX}_${model_slug}_${VECTOR_SUFFIX}.svec"
  fi

  # Baseline cache: include subset so a tiny test run doesn't get reused for a full run
  local baseline_cache="mcq_cache/baseline_${DATASET_SLUG}_${model_slug}_k${FEWSHOT_K}_seed${SEED}_es${EVAL_START}_el${EVAL_LIMIT}.json"

  echo "======================================================"
  echo "Running A/B APR MCQ eval"
  echo "  Dataset       : ${DATASET_SLUG}"
  echo "  MCQ file      : ${MCQ_FILE}"
  echo "  Model         : ${model_id}"
  echo "  Tokenizer     : ${tokenizer_id}"
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

if [[ "$RUN_CODELLAMA" -eq 1 ]]; then
  run_eval "${CODELLAMA_MODEL}" "${CODELLAMA_TOKENIZER}" "${CODELLAMA_SLUG}"
fi

if [[ "$RUN_QWEN_INST_7B" -eq 1 ]]; then
  run_eval "${QWEN_INST_7B_MODEL}" "${QWEN_INST_7B_TOKENIZER}" "${QWEN_INST_7B_SLUG}"
fi

if [[ "$RUN_QWEN_CODER_7B" -eq 1 ]]; then
  run_eval "${QWEN_CODER_7B_MODEL}" "${QWEN_CODER_7B_TOKENIZER}" "${QWEN_CODER_7B_SLUG}"
fi

if [[ "$RUN_QWEN_CODER_14B" -eq 1 ]]; then
  run_eval "${QWEN_CODER_14B_MODEL}" "${QWEN_CODER_14B_TOKENIZER}" "${QWEN_CODER_14B_SLUG}"
fi

