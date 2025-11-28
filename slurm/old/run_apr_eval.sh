#!/bin/bash

#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --gres=gpu:a100
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
nvidia-smi
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0
cd activation-steering#!/bin/bash

#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --gres=gpu:a100
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
nvidia-smi
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0
cd activation-steering-orig/




#python demo-extract.py

python ab_apr_eval.py \
  --pairs_path /lustre/hdd/LAS/jannesar-lab/arushi/activation-steering-orig/docs/demo-data/filtered-2.jsonl \
  --start 0 \
  --fewshot_k 3 \
  --model_id meta-llama/CodeLlama-7b-Instruct-hf\
  --compare \
  --vector_path refusal_behavior_vector-codellama\
  --strength 2 \
  --layers 22,23,24,25,26,27
#Qwen/Qwen2.5-Coder-7B-Instruct meta-llama/CodeLlama-7b-Instruct-hf /lustre/hdd/LAS/jannesar-lab/arushi/activation-steering-orig/finetuning/qwen2_5_tssb_qlora
