#!/bin/bash

#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --job-name="apr steering"
#SBATCH --mail-user=arushi17@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="logs/slurm-%j.out"


export MPLCONFIGDIR=/lustre/hdd/LAS/jannesar-lab/arushi/matplotlib
export XDG_CACHE_HOME=/lustre/hdd/LAS/jannesar-lab/arushi/cache
export TRITON_CACHE_DIR=/lustre/hdd/LAS/jannesar-lab/arushi/cache/triton
export HF_HOME=/lustre/hdd/LAS/jannesar-lab/arushi/
cd /lustre/hdd/LAS/jannesar-lab/arushi
source myenv/bin/activate
export CUDA_VISIBLE_DEVICES=0
cd activation-steering-orig/


python demo-extract.py

python ab_apr_eval.py \
  --pairs_path /lustre/hdd/LAS/jannesar-lab/arushi/tssb_data_3M/filtered-7.jsonl \
  --start 0 --limit 100 \
  --fewshot_k 3 \
  --model_id meta-llama/CodeLlama-7b-Instruct-hf \
  --compare \
  --vector_path refusal_behavior_vector \
  --strength 2.0 \
  --layers 27,28,29,30,31

