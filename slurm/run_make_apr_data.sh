#!/bin/bash

#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --job-name="apr_data_build"
#SBATCH --mail-user=arushi17@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="logs/slurm-%j.out"

########################################
# Environment / Setup
########################################

export HF_HOME=/lustre/hdd/LAS/jannesar-lab/arushi/
export HF_CACHE=/lustre/hdd/LAS/jannesar-lab/arushi/activation-steering/models
export TRANSFORMERS_CACHE=/lustre/hdd/LAS/jannesar-lab/arushi/activation-steering/models

cd /lustre/hdd/LAS/jannesar-lab/arushi
source myenv/bin/activate

cd activation-steering

########################################
# Paths
########################################

DERIVED="data/tssb_data_3M/derived"
FILTERED="${DERIVED}/filtered-0.jsonl"

APR_DATA_SCRIPT="${DERIVED}/apr_data.py"
APR_QUESTIONS_SCRIPT="${DERIVED}/make_apr_questions.py"

########################################
# Step 1 — Create correctness contrastive pairs
########################################

python "${APR_DATA_SCRIPT}" \
  "${FILTERED}" \
  --out "${DERIVED}/correctness_behavior_apr.json" \
  --num_samples 2000

########################################
# Step 2 — Make APR MCQ questions (OPTIONAL)
# Commented out for now
########################################

# python "${APR_QUESTIONS_SCRIPT}" \
#   "${FILTERED}" \
#   --out "${DERIVED}/apr_questions_2k_5k.json" \
#   --start 2000 \
#   --end 5000

echo "APR core data build complete."

