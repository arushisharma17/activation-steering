#!/bin/bash

#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:a100
#SBATCH --mem=128G
#SBATCH --job-name="mcq_sweep_injob"
#SBATCH --mail-user=arushi17@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="logs/slurm-mcq-sweep-%j.out"

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
# Defaults
########################################

DATASET="tssb"          # tssb | manysstubs
FEWSHOT_K=3
SEED=42
MCQ_FILE=""             # if empty -> mcq_cache/<dataset>_mcq_kK_seedS.json

# Eval subset controls (pass through to ab_apr_eval via run_mcq_eval.sh)
EVAL_START=0            # index of first eval item
EVAL_LIMIT=5000            # 0 = all items from EVAL_START

# Which models to sweep inside this job
# Parsed from --models "all" or space-separated list:
#   codellama-7b qwen-inst-7b qwen-coder-7b qwen-coder-14b
MODELS_STR="all"

# Layer bands (fractional depth) and strengths to sweep
LAYER_BANDS=("band:0.15:6" "band:0.35:6" "band:0.55:6" "band:0.75:6" "band:0.90:6")
STRENGTHS=(1.0 1.5 2.0 2.5 3.0)

########################################
# Parse CLI args
########################################

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
    --mcq-file)
      MCQ_FILE="$2"
      shift 2
      ;;
    --models)
      MODELS_STR="$2"   # e.g. "all" or "qwen-coder-7b" or "codellama-7b qwen-coder-7b"
      shift 2
      ;;
    --layers)
      # override full list: pass a quoted string like "band:0.55:6 band:0.75:6"
      IFS=' ' read -r -a LAYER_BANDS <<< "$2"
      shift 2
      ;;
    --strengths)
      # override strengths: pass "1.0 1.5 2.0"
      IFS=' ' read -r -a STRENGTHS <<< "$2"
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
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

########################################
# Dataset → vector dir + naming
########################################

VECTOR_SUFFIX="pca_center_suffix"

case "$DATASET" in
  tssb)
    # Python-based TSSB vectors
    VECTOR_DIR="vectors/python"
    DATASET_SLUG="tssb"
    PAIRS_STEM_SLUG="correctness_behavior_apr"
    ;;
  manysstubs)
    # Java-based ManySStuBs4J vectors
    VECTOR_DIR="vectors/java"
    DATASET_SLUG="manysstubs"
    PAIRS_STEM_SLUG="apr_manysstubs_pairs_3k"
    ;;
  *)
    echo "ERROR: --dataset must be 'tssb' or 'manysstubs' (got '$DATASET')."
    exit 1
    ;;
esac

########################################
# Model IDs and slugs (match vector names)
########################################

# 1) CodeLlama-7B-Instruct
CODELLAMA_SLUG="codellama-7b-instruct-hf"

# 2) Qwen2.5-7B-Instruct
QWEN_INST_7B_SLUG="qwen2-5-7b-instruct"

# 3) Qwen2.5-Coder-7B-Instruct
QWEN_CODER_7B_SLUG="qwen2-5-coder-7b-instruct"

# 4) Qwen2.5-Coder-14B-Instruct
QWEN_CODER_14B_SLUG="qwen2-5-coder-14b-instruct"

########################################
# Enable models from --models
########################################

RUN_CODELLAMA=0
RUN_QWEN_INST_7B=0
RUN_QWEN_CODER_7B=0
RUN_QWEN_CODER_14B=0

if [[ "$MODELS_STR" == "all" ]]; then
  RUN_CODELLAMA=1
  RUN_QWEN_INST_7B=1
  RUN_QWEN_CODER_7B=1
  RUN_QWEN_CODER_14B=1
else
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
fi

########################################
# Resolve MCQ file path
########################################

if [[ -z "$MCQ_FILE" ]]; then
  MCQ_FILE="mcq_cache/${DATASET}_mcq_k${FEWSHOT_K}_seed${SEED}.json"
fi

if [[ ! -f "$MCQ_FILE" ]]; then
  echo "ERROR: MCQ file not found: ${MCQ_FILE}"
  echo "       Build it first with slurm/run_mcq_build.sh."
  exit 1
fi

########################################
# Helper: check steering vector exists
########################################

check_vector () {
  local model_slug="$1"
  local vec_path="${VECTOR_DIR}/behavior_${DATASET_SLUG}_${PAIRS_STEM_SLUG}_${model_slug}_${VECTOR_SUFFIX}.svec"
  if [[ ! -f "$vec_path" ]]; then
    echo "[WARN] Steering vector not found for ${model_slug} @ ${vec_path}"
    echo "       Skipping this model for dataset=${DATASET}."
    return 1
  fi
  return 0
}

# Drop models that lack vectors
if [[ "$RUN_CODELLAMA" -eq 1 ]]; then
  if ! check_vector "$CODELLAMA_SLUG"; then RUN_CODELLAMA=0; fi
fi
if [[ "$RUN_QWEN_INST_7B" -eq 1 ]]; then
  if ! check_vector "$QWEN_INST_7B_SLUG"; then RUN_QWEN_INST_7B=0; fi
fi
if [[ "$RUN_QWEN_CODER_7B" -eq 1 ]]; then
  if ! check_vector "$QWEN_CODER_7B_SLUG"; then RUN_QWEN_CODER_7B=0; fi
fi
if [[ "$RUN_QWEN_CODER_14B" -eq 1 ]]; then
  if ! check_vector "$QWEN_CODER_14B_SLUG"; then RUN_QWEN_CODER_14B=0; fi
fi

if [[ "$RUN_CODELLAMA" -eq 0 && "$RUN_QWEN_INST_7B" -eq 0 && "$RUN_QWEN_CODER_7B" -eq 0 && "$RUN_QWEN_CODER_14B" -eq 0 ]]; then
  echo "[INFO] No models to run (no steering vectors found). Exiting."
  exit 0
fi

########################################
# Summary of what we'll run
########################################

echo "MCQ steering sweep (single Slurm job)"
echo "  Dataset    : ${DATASET}"
echo "  MCQ file   : ${MCQ_FILE}"
echo "  Few-shot k : ${FEWSHOT_K}"
echo "  Seed       : ${SEED}"
echo "  Eval start : ${EVAL_START}"
echo "  Eval limit : ${EVAL_LIMIT}"
echo -n "  Models     :"
[[ "$RUN_CODELLAMA"    -eq 1 ]] && echo -n " codellama-7b"
[[ "$RUN_QWEN_INST_7B" -eq 1 ]] && echo -n " qwen-inst-7b"
[[ "$RUN_QWEN_CODER_7B" -eq 1 ]] && echo -n " qwen-coder-7b"
[[ "$RUN_QWEN_CODER_14B" -eq 1 ]] && echo -n " qwen-coder-14b"
echo
echo -n "  Layer bands:"
for L in "${LAYER_BANDS[@]}"; do echo -n " ${L}"; done
echo
echo -n "  Strengths  :"
for A in "${STRENGTHS[@]}"; do echo -n " ${A}"; done
echo
echo "============================================"

########################################
# Helper: run sweep for one model
########################################

run_sweep_for_model () {
  local model_label="$1"   # e.g. "qwen-coder-7b"
  local model_flag="$2"    # e.g. "--qwen-coder-7b"

  echo "--------------------------------------------"
  echo "[MODEL] ${model_label}"
  echo "--------------------------------------------"

  for LAY in "${LAYER_BANDS[@]}"; do
    for A in "${STRENGTHS[@]}"; do
      echo ">>> ${model_label} | layers=${LAY} | strength=${A}"
      # Important: call as plain bash script, NOT sbatch
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
# Run selected models
########################################

if [[ "$RUN_CODELLAMA" -eq 1 ]]; then
  run_sweep_for_model "codellama-7b" "--codellama-7b"
fi

if [[ "$RUN_QWEN_INST_7B" -eq 1 ]]; then
  run_sweep_for_model "qwen-inst-7b" "--qwen-inst-7b"
fi

if [[ "$RUN_QWEN_CODER_7B" -eq 1 ]]; then
  run_sweep_for_model "qwen-coder-7b" "--qwen-coder-7b"
fi

if [[ "$RUN_QWEN_CODER_14B" -eq 1 ]]; then
  run_sweep_for_model "qwen-coder-14b" "--qwen-coder-14b"
fi

echo "[INFO] Sweep finished."

