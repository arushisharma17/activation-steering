#!/bin/bash

#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=200G
#SBATCH --job-name="hecbench_bugs"
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

cd /lustre/hdd/LAS/jannesar-lab/rnahra

cd activation-steering

########################################
# LLM configuration (local model)
# Override these as needed
########################################

export LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8000/v1}"
export LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
export LLM_MODEL="/lustre/hdd/LAS/jannesar-lab/rnahra/gpt-oss-120b"
export LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.7}"
export LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-4096}"

########################################
# Paths
########################################

PIPELINE_DIR="data/hecbench"
REPO_DIR="/tmp/hecbench_repo"

export PYTHON_EXEC="/lustre/hdd/LAS/jannesar-lab/rnahra/conda_envs/steer/bin/python"

########################################
# Step 1 — Fetch HeCBench sources
########################################

echo "=== Step 1: Fetching HeCBench sources ==="

${PYTHON_EXEC} "${PIPELINE_DIR}/fetch_hecbench_sources.py" \
  --repo-dir "${REPO_DIR}" \
  --out "${PIPELINE_DIR}/sources"

########################################
# Step 2 — Inject bugs via LLM
########################################

echo "=== Step 2: Injecting bugs via LLM ==="

${PYTHON_EXEC} "${PIPELINE_DIR}/inject_bugs_llm.py" \
  --manifest "${PIPELINE_DIR}/sources/manifest.json" \
  --sources-dir "${PIPELINE_DIR}/sources" \
  --out "${PIPELINE_DIR}/raw_bugs.jsonl"

# To limit the run to specific bug types or a max number of samples:
#   --bug-types RACE_CONDITION,MISSING_ATOMIC
#   --max-samples 200

########################################
# Step 3 — Build contrastive pairs
########################################

echo "=== Step 3: Building contrastive pairs ==="

${PYTHON_EXEC} "${PIPELINE_DIR}/build_contrastive_pairs.py" \
  --input "${PIPELINE_DIR}/raw_bugs.jsonl" \
  --out "${PIPELINE_DIR}/contrastive_pairs_hecbench.json" \
  --by-category

echo "=== Pipeline complete ==="
