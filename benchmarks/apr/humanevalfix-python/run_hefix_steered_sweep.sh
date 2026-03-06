#!/bin/bash
#
# Script: run_hefix_steered_strength_sweep.sh
#
# HumanEvalFix-Python multi-model runner (STEERED ONLY)
# Sweeps multiple strengths per model.
#

#SBATCH --time=5-23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --job-name="hefix-steer-sweep"
#SBATCH --mail-user=arushi17@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="logs/slurm-%j.out"

set -euo pipefail

########################################
# Paths (define ROOT before exports!)
########################################
ROOT="/lustre/hdd/LAS/jannesar-lab/arushi/activation-steering"
cd "$ROOT/benchmarks/apr/humanevalfix-python"

mkdir -p logs
mkdir -p hefix_generations

########################################
# GLOBAL — SET TOKEN FIRST
########################################
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export HF_HUB_ENABLE_HF_TRANSFER="0"

# Caches (Nova-style)
export MPLCONFIGDIR="$ROOT/matplotlib"
export XDG_CACHE_HOME="$ROOT/cache"
export TRITON_CACHE_DIR="$ROOT/cache/triton"
export HF_HOME="$ROOT/models"

export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

echo "[INFO] Running on node: $(hostname)"
echo "[INFO] Working dir: $(pwd)"
nvidia-smi || echo "[WARN] nvidia-smi not available"

########################################
# Defaults
########################################
DATASET="tssb"          # tssb | manysstubs
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

# strength sweep (comma-separated)
# You can override via --strengths "0.5,1.0,1.5,2.0"
STRENGTHS="0.5,1.0,1.5,2.0,2.5,3.0"

########################################
# Parse CLI arguments
########################################
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)  DATASET="$2"; shift 2 ;;
    --eval-only) EVAL_ONLY=true; shift ;;

    # gen knobs
    --n) N_SAMPLES="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --top-p) TOP_P="$2"; shift 2 ;;
    --max-tries) MAX_TRIES="$2"; shift 2 ;;
    --min-chars) MIN_CHARS="$2"; shift 2 ;;
    --retry-temp-mult) RETRY_TEMP_MULT="$2"; shift 2 ;;
    --no-overwrite) NO_OVERWRITE=true; shift ;;

    # eval
    --max-k) MAX_K="$2"; shift 2 ;;
    --k-list) K_LIST="$2"; shift 2 ;;

    # strength sweep
    --strengths) STRENGTHS="$2"; shift 2 ;;

    # model subsets
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
if [[ "$DATASET" != "tssb" && "$DATASET" != "manysstubs" ]]; then
  echo "[ERROR] --dataset must be 'tssb' or 'manysstubs' (got '$DATASET')."
  exit 1
fi

if [ "$EXPLICIT_MODELS" -eq 0 ]; then
  RUN_QWEN_INST_7B=true
  RUN_QWEN_CODER_7B=true
  RUN_QWEN_CODER_14B=true
  RUN_CODELLAMA_7B=true
fi

########################################
# Build eval k list
########################################
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

VEC_ROOT="${ROOT}/vectors"
if [[ "$DATASET" == "tssb" ]]; then
  echo "[INFO] Using TSSB-trained steering vectors (correctness_vector_100)."
  VECTORS=(
    "${VEC_ROOT}/python/correctness_vector_100_qwen2-5-7b-instruct_pca_center_suffix.svec"
    "${VEC_ROOT}/python/correctness_vector_100_qwen2-5-coder-7b-instruct_pca_center_suffix.svec"
    "${VEC_ROOT}/python/correctness_vector_100_qwen2-5-coder-14b-instruct_pca_center_suffix.svec"
    "${VEC_ROOT}/python/correctness_vector_100_codellama-7b-instruct-hf_pca_center_suffix.svec"
  )
  VEC_IDS=("tssb-qwen7b-general" "tssb-qwencoder7b" "tssb-qwencoder14b" "tssb-codellama7b")
else
  echo "[INFO] Using ManySStuBs-trained steering vectors."
  VECTORS=(
    "${VEC_ROOT}/java/behavior_manysstubs_apr_manysstubs_pairs_3k_qwen2-5-7b-instruct_pca_center_suffix.svec"
    "${VEC_ROOT}/java/behavior_manysstubs_apr_manysstubs_pairs_3k_qwen2-5-coder-7b-instruct_pca_center_suffix.svec"
    "${VEC_ROOT}/java/behavior_manysstubs_apr_manysstubs_pairs_3k_qwen2-5-coder-14b-instruct_pca_center_suffix.svec"
    "${VEC_ROOT}/java/behavior_manysstubs_apr_manysstubs_pairs_3k_codellama-7b-instruct-hf_pca_center_suffix.svec"
  )
  VEC_IDS=("manysstubs-qwen7b-general" "manysstubs-qwencoder7b" "manysstubs-qwencoder14b" "manysstubs-codellama7b")
fi

N_MODELS=${#MODELS[@]}

# Keep your best layers per model
BEST_LAYERS=("band:0.55:6" "band:0.15:6" "band:0.15:6" "band:0.75:6")

########################################
# Activate env
########################################
source /lustre/hdd/LAS/jannesar-lab/arushi/myenv/bin/activate

run_eval () {
  local MODEL_SLUG=$1
  local CONDITION=$2
  python eval_hefix.py --model_slug "$MODEL_SLUG" --condition "$CONDITION" --k_list "$KCSV"
}

########################################
# Gen flags bundle
########################################
GEN_FLAGS=(
  --n "$N_SAMPLES"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P"
  --max_tries "$MAX_TRIES"
  --min_chars "$MIN_CHARS"
  --retry_temp_mult "$RETRY_TEMP_MULT"
)
if [ "$NO_OVERWRITE" = true ]; then
  GEN_FLAGS+=(--no_overwrite)
fi

########################################
# Parse strengths list
########################################
IFS=',' read -r -a STRENGTH_ARR <<< "$STRENGTHS"

echo "[INFO] CONFIG:"
echo "  DATASET         = $DATASET"
echo "  EVAL_ONLY       = $EVAL_ONLY"
echo "  STRENGTHS       = ${STRENGTH_ARR[*]}"
echo "  N_SAMPLES       = $N_SAMPLES"
echo "  MAX_NEW_TOKENS  = $MAX_NEW_TOKENS"
echo "  TEMP / TOP_P    = $TEMPERATURE / $TOP_P"
echo "  EVAL_KS         = ${EVAL_KS[*]}"
echo ""

########################################
# STEERED (strength sweep)
########################################
echo "[INFO] === STEERED strength sweep (dataset=$DATASET) ==="
TS=$(date +"%Y%m%d-%H%M")

for ((i=0; i<"$N_MODELS"; i++)); do
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

  HP_LAYERS_TAG="L${LAYERS//:/-}"

  for STRENGTH in "${STRENGTH_ARR[@]}"; do
    # tag strength safely in filenames/condition
    HP_STRENGTH_TAG="a${STRENGTH//./p}"   # 2.5 -> a2p5

    CONDITION="steer-${VEC_ID}_${HP_LAYERS_TAG}_${HP_STRENGTH_TAG}_${TS}"

    echo
    echo "[INFO] --- STEERED: $MODEL_ID ($MODEL_SLUG) ---"
    echo "[INFO] vec_path=$VEC_PATH"
    echo "[INFO] layers=$LAYERS strength=$STRENGTH"
    echo "[INFO] condition=$CONDITION"

    if [ "$EVAL_ONLY" = false ]; then
      python gen_hefix.py \
        --model "$MODEL_ID" \
        --condition "$CONDITION" \
        --steer \
        --vector_path "$VEC_PATH" \
        --layers "$LAYERS" \
        --strength "$STRENGTH" \
        "${GEN_FLAGS[@]}"
    else
      echo "[INFO] EVAL_ONLY=true -> skipping generation for $MODEL_SLUG / $STRENGTH"
    fi

    run_eval "$MODEL_SLUG" "$CONDITION"
  done
done

echo
echo "[INFO] Aggregating summaries..."
python summarize_hefix_results.py || echo "[WARN] summarize_hefix_results.py failed"

deactivate
echo "[INFO] DONE."

