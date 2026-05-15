#!/bin/bash
#SBATCH --account=f2025.coms.5990.01
#SBATCH --partition=instruction 
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --job-name="apr_eval"
#SBATCH --mail-user=moulica9@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="logs/slurm-%j.out"
#SBATCH --error="logs/slurm-%j.err"

# Exit on any error
set -e
set -x  # Print each command (for debugging)

MODEL_ID=${1:-"Qwen/Qwen2.5-Coder-7B-Instruct"}
PAIRS_PATH=${2:-"/work/classtmp/moulica9/activation-steering/humaneval_ab_format.jsonl"}
LAYERS=${3:-"25,26,27,28,29"}
STRENGTH=${4:-"2.0"}
FEWSHOT_K=${5:-"5"}

echo "========================================="
echo "Starting APR Evaluation"
echo "Time: $(date)"
echo "========================================="

cd /work/classtmp/moulica9/activation-steering || exit 1
source myenv/bin/activate || exit 1

export HF_TOKEN= ######
export CUDA_VISIBLE_DEVICES=0

# Verify files exist
if [ ! -f "$PAIRS_PATH" ]; then
    echo "ERROR: Pairs file not found: $PAIRS_PATH"
    exit 1
fi

if [ ! -f "demo-extract.py" ]; then
    echo "ERROR: demo-extract.py not found"
    exit 1
fi

if [ ! -f "ab_apr_eval.py" ]; then
    echo "ERROR: ab_apr_eval.py not found"
    exit 1
fi

# Create unique folder name for this run
MODEL_NAME=$(echo "$MODEL_ID" | sed 's/\//_/g' | sed 's/-/_/g')
PAIRS_NAME=$(basename "$PAIRS_PATH" .jsonl)
LAYERS_CLEAN=$(echo "$LAYERS" | sed 's/,/_/g')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results/run_${MODEL_NAME}_${PAIRS_NAME}_layers_${LAYERS_CLEAN}_str_${STRENGTH}_k_${FEWSHOT_K}_${TIMESTAMP}"

mkdir -p "$OUTPUT_DIR"

echo "========================================" | tee "$OUTPUT_DIR/config.txt"
echo "Model: $MODEL_ID" | tee -a "$OUTPUT_DIR/config.txt"
echo "Pairs: $PAIRS_PATH" | tee -a "$OUTPUT_DIR/config.txt"
echo "Layers: $LAYERS" | tee -a "$OUTPUT_DIR/config.txt"
echo "Strength: $STRENGTH" | tee -a "$OUTPUT_DIR/config.txt"
echo "Fewshot K: $FEWSHOT_K" | tee -a "$OUTPUT_DIR/config.txt"
echo "Output: $OUTPUT_DIR" | tee -a "$OUTPUT_DIR/config.txt"
echo "========================================" | tee -a "$OUTPUT_DIR/config.txt"

# Generate steering vector
echo "" | tee -a "$OUTPUT_DIR/config.txt"
echo "Step 1: Generating steering vector..." | tee -a "$OUTPUT_DIR/config.txt"
echo "Start time: $(date)" | tee -a "$OUTPUT_DIR/config.txt"

python demo-extract.py \
    --model_id "$MODEL_ID" \
    --vector_name "${OUTPUT_DIR}/refusal_behavior_vector" 2>&1 | tee "$OUTPUT_DIR/extract.log"

EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: demo-extract.py failed with exit code $EXIT_CODE" | tee -a "$OUTPUT_DIR/config.txt"
    exit $EXIT_CODE
fi

echo "Vector generation completed: $(date)" | tee -a "$OUTPUT_DIR/config.txt"

# FIXED: Check for .svec file (the actual output format)
if [ ! -f "${OUTPUT_DIR}/refusal_behavior_vector.svec" ]; then
    echo "ERROR: Steering vector not created at ${OUTPUT_DIR}/refusal_behavior_vector.svec" | tee -a "$OUTPUT_DIR/config.txt"
    echo "Files actually created:" | tee -a "$OUTPUT_DIR/config.txt"
    ls -lah "${OUTPUT_DIR}/" | tee -a "$OUTPUT_DIR/config.txt"
    exit 1
fi

VECTOR_SIZE=$(du -h "${OUTPUT_DIR}/refusal_behavior_vector.svec" | cut -f1)
echo "✓ Steering vector verified (${VECTOR_SIZE})" | tee -a "$OUTPUT_DIR/config.txt"

# Run evaluation
echo "" | tee -a "$OUTPUT_DIR/config.txt"
echo "Step 2: Running evaluation..." | tee -a "$OUTPUT_DIR/config.txt"
echo "Start time: $(date)" | tee -a "$OUTPUT_DIR/config.txt"

python ab_apr_eval.py \
    --pairs_path "$PAIRS_PATH" \
    --start 0 --limit 100 \
    --fewshot_k $FEWSHOT_K \
    --model_id "$MODEL_ID" \
    --compare \
    --vector_path "${OUTPUT_DIR}/refusal_behavior_vector" \
    --strength $STRENGTH \
    --layers "$LAYERS" \
    --show_n 3 2>&1 | tee "$OUTPUT_DIR/output.log"

EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: ab_apr_eval.py failed with exit code $EXIT_CODE" | tee -a "$OUTPUT_DIR/config.txt"
    exit $EXIT_CODE
fi

echo "Evaluation completed: $(date)" | tee -a "$OUTPUT_DIR/config.txt"
echo "" | tee -a "$OUTPUT_DIR/config.txt"
echo "========================================" | tee -a "$OUTPUT_DIR/config.txt"
echo "✓ All steps completed successfully" | tee -a "$OUTPUT_DIR/config.txt"
echo "✓ Results saved to: $OUTPUT_DIR" | tee -a "$OUTPUT_DIR/config.txt"
echo "End time: $(date)" | tee -a "$OUTPUT_DIR/config.txt"
echo "========================================" | tee -a "$OUTPUT_DIR/config.txt"
