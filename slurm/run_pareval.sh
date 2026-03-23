#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200G
#SBATCH --job-name="pareval_steer"
#SBATCH --mail-user=rnahra@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="logs/slurm-%j.out"

########################################
# Environment / Setup
########################################

export HF_HOME=/lustre/hdd/LAS/jannesar-lab/rnahra/
export HF_CACHE=/lustre/hdd/LAS/jannesar-lab/rnahra/activation-steering/models
export TRANSFORMERS_CACHE=/lustre/hdd/LAS/jannesar-lab/rnahra/activation-steering/models

cd /lustre/hdd/LAS/jannesar-lab/rnahra/activation-steering

export PYTHON_EXEC="/lustre/hdd/LAS/jannesar-lab/rnahra/conda_envs/steer/bin/python"

########################################
# Configuration — edit these as needed
########################################

MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
VECTOR="vectors/python/correctness_vector_100_qwen2-5-coder-7b-instruct_pca_center_suffix"
STRENGTH=1.5
LAYERS="15,16,17,18,19,20,21,22,23"
PROMPTS="ParEval/prompts/generation-prompts.json"
NUM_SAMPLES=2
MAX_PROMPTS=5
INCLUDE_MODELS="omp"
TEMPERATURE=0.2

OUTPUT_DIR="results/pareval"
mkdir -p "${OUTPUT_DIR}"

########################################
# Step 1 — Steered run
########################################

echo "=== Step 1: Steered ParEval run ==="

${PYTHON_EXEC} scripts/run_pareval.py \
  --prompts "${PROMPTS}" \
  --model "${MODEL}" \
  --vector_path "${VECTOR}" \
  --strength ${STRENGTH} \
  --layers "${LAYERS}" \
  --output "${OUTPUT_DIR}/steered_outputs.json" \
  --cache "${OUTPUT_DIR}/steered_cache.jsonl" \
  --num_samples_per_prompt ${NUM_SAMPLES} \
  --include_models "${INCLUDE_MODELS}" \
  --max_prompts ${MAX_PROMPTS} \
  --temperature ${TEMPERATURE} \
  --do_sample \
  --use_chat_template

########################################
# Step 2 — Baseline run (no steering)
########################################

echo "=== Step 2: Baseline ParEval run ==="

${PYTHON_EXEC} scripts/run_pareval.py \
  --prompts "${PROMPTS}" \
  --model "${MODEL}" \
  --no_steer \
  --output "${OUTPUT_DIR}/baseline_outputs.json" \
  --cache "${OUTPUT_DIR}/baseline_cache.jsonl" \
  --num_samples_per_prompt ${NUM_SAMPLES} \
  --include_models "${INCLUDE_MODELS}" \
  --max_prompts ${MAX_PROMPTS} \
  --temperature ${TEMPERATURE} \
  --do_sample \
  --use_chat_template

echo "=== Generation complete ==="
echo "Steered output: ${OUTPUT_DIR}/steered_outputs.json"
echo "Baseline output: ${OUTPUT_DIR}/baseline_outputs.json"
echo ""
echo "Next steps — run ParEval evaluation:"
echo "  cd ParEval/drivers"
echo "  python run-all.py ../../${OUTPUT_DIR}/steered_outputs.json -o ../../${OUTPUT_DIR}/steered_results.json --yes-to-all --include-models ${INCLUDE_MODELS}"
echo "  python run-all.py ../../${OUTPUT_DIR}/baseline_outputs.json -o ../../${OUTPUT_DIR}/baseline_results.json --yes-to-all --include-models ${INCLUDE_MODELS}"
