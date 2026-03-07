#!/bin/bash
#
# Usage examples:
#   sbatch run_codellama_hejava.sh                 # baseline + steered (gen + eval)
#   sbatch run_codellama_hejava.sh --baseline      # baseline only (gen + eval)
#   sbatch run_codellama_hejava.sh --steered       # steered only (gen + eval)
#   sbatch run_codellama_hejava.sh --baseline --eval-only   # eval baseline only
#   sbatch run_codellama_hejava.sh --steered  --eval-only   # eval latest steered only
#
#SBATCH --time=2-23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --gres=gpu:a100
#SBATCH --mem=128G
#SBATCH --job-name="hejava-codellama"
#SBATCH --mail-user=arushi17@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="logs/slurm-codellama-%j.out"

########################################
# Parse CLI arguments
########################################

RUN_BASELINE=false
RUN_STEERED=false
EVAL_ONLY=false

# If no args passed, default = run both (gen + eval)
if [ $# -eq 0 ]; then
  RUN_BASELINE=true
  RUN_STEERED=true
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline)
      RUN_BASELINE=true
      shift
      ;;
    --steered)
      RUN_STEERED=true
      shift
      ;;
    --eval-only)
      EVAL_ONLY=true
      shift
      ;;
    *)
      echo "[ERROR] Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "[INFO] CONFIG:"
echo "  RUN_BASELINE = $RUN_BASELINE"
echo "  RUN_STEERED  = $RUN_STEERED"
echo "  EVAL_ONLY    = $EVAL_ONLY"
echo ""

########################################
# Environment setup
########################################

export HF_HOME=/lustre/hdd/LAS/jannesar-lab/arushi
export HF_CACHE=/lustre/hdd/LAS/jannesar-lab/arushi/activation-steering/models
export TRANSFORMERS_CACHE=/lustre/hdd/LAS/jannesar-lab/arushi/activation-steering/models
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

# Maven local repo on Lustre
export MAVEN_REPO=/lustre/hdd/LAS/jannesar-lab/arushi/maven-repo
mkdir -p "$MAVEN_REPO"
chmod -R u+rwX "$MAVEN_REPO"

# Java + Maven modules
module purge
module load openjdk/11.0.17_8-ixtfxgw
module load maven/3.8.4-ieg7dba

# Go directly to the HE-Java benchmark inside activation-steering
cd /lustre/hdd/LAS/jannesar-lab/arushi/activation-steering/benchmarks/apr/human-eval-java
mkdir -p logs

echo "[INFO] Running on node: $(hostname)"
echo "[INFO] Working dir: $(pwd)"
nvidia-smi || echo "[WARN] nvidia-smi not available"

########################################
# Activate env once
########################################
source /lustre/hdd/LAS/jannesar-lab/arushi/myenv/bin/activate

########################################
# Model + steering config (CodeLlama only)
########################################

MODEL_ID="meta-llama/CodeLlama-7b-Instruct-hf"
MODEL_SLUG="CodeLlama-7b-Instruct-hf"

VEC_PATH="/lustre/hdd/LAS/jannesar-lab/arushi/activation-steering/vectors/python/refusal_behavior_vector-codellama.svec"
VEC_ID="codellama7b"   # used for naming steer-* dirs

########################################
# Helper: run eval for k = 1,5,10
########################################
run_eval_for_condition () {
  local MODEL_SLUG=$1
  local CONDITION=$2

  echo "[INFO] Evaluating condition '$CONDITION' for model '$MODEL_SLUG'..."

  # pass@1
  python meta/scripts/eval_hejava.py \
    --model_slug "$MODEL_SLUG" \
    --condition "$CONDITION" \
    --max_k 1

  # pass@5
  python meta/scripts/eval_hejava.py \
    --model_slug "$MODEL_SLUG" \
    --condition "$CONDITION" \
    --max_k 5

  # pass@10
  python meta/scripts/eval_hejava.py \
    --model_slug "$MODEL_SLUG" \
    --condition "$CONDITION" \
    --max_k 10
}

########################################
# 1) BASELINE (no steering)
########################################

if [ "$RUN_BASELINE" = true ]; then
  echo
  echo "[INFO] === BASELINE: $MODEL_ID ($MODEL_SLUG) ==="

  if [ "$EVAL_ONLY" = false ]; then
    echo "[INFO] Generating baseline completions (n=10)..."
    python meta/scripts/gen_hejava.py \
      --model "$MODEL_ID" \
      --n 10

    echo "[INFO] Baseline generation done for $MODEL_SLUG."
  else
    echo "[INFO] Skipping baseline generation (EVAL_ONLY=true)."
  fi

  echo "[INFO] Running baseline evaluation..."
  run_eval_for_condition "$MODEL_SLUG" "baseline"
fi

########################################
# 2) STEERED
########################################

if [ "$RUN_STEERED" = true ]; then
  echo
  echo "[INFO] === STEERED: $MODEL_ID ($MODEL_SLUG) ==="
  echo "[INFO] Using steering vector: $VEC_PATH (vec_id=$VEC_ID)"

  if [ "$EVAL_ONLY" = false ]; then
    python meta/scripts/gen_hejava.py \
      --model "$MODEL_ID" \
      --n 10 \
      --steer \
      --vector_path "$VEC_PATH" \
      --strength 2.0 \
      --layers "last:4" \
      --vec_id "$VEC_ID"

    echo "[INFO] Steered generation done for $MODEL_SLUG."
  else
    echo "[INFO] Skipping steered generation (EVAL_ONLY=true)."
  fi

  # Auto-detect newest steer-* directory for this model
  STEER_COND=$(ls -td "hej_generations/$MODEL_SLUG"/steer-* | head -n 1 | xargs -n 1 basename)
  echo "[INFO] Auto-detected steered condition for $MODEL_SLUG: $STEER_COND"

  echo "[INFO] Running steered evaluation..."
  run_eval_for_condition "$MODEL_SLUG" "$STEER_COND"
fi

deactivate

echo "[INFO] All done (CodeLlama HE-Java)."

