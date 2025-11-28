#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --gres=gpu:a100
#SBATCH --mem=40G
#SBATCH --job-name="apr steering"
#SBATCH --mail-user=arushi17@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="logs/slurm-%j.out"


export MPLCONFIGDIR=/lustre/hdd/LAS/jannesar-lab/arushi/matplotlib
export XDG_CACHE_HOME=/lustre/hdd/LAS/jannesar-lab/arushi/cache
export HF_HOME=/lustre/hdd/LAS/jannesar-lab/arushi/
cd /lustre/hdd/LAS/jannesar-lab/arushi
source myenv/bin/activate
export HUGGINGFACE_TOKEN=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
export CUDA_VISIBLE_DEVICES=0
cd activation-steering-orig/


python demo-condition.py

python ab_eval_multisteer.py \
  --pairs_path /lustre/hdd/LAS/jannesar-lab/arushi/activation-steering-orig/docs/demo-data/filtered-2.jsonl \
  --mode steered \
  --compare \
  --model_id Qwen/Qwen2.5-Coder-7B-Instruct\
  --vector_path refusal_behavior_vector-qwen2 \
  --behavior_stem refusal_behavior_vector-qwen2 \
  --cond_dir condition_vectors-qwen2 \
  --behavior_layers 18,19,20,21,22,23 \
  --strength 1.5 \
  --rule_mode any \
# meta-llama/CodeLlama-7b-Instruct-hf Qwen/Qwen2.5-Coder-7B-Instruct
