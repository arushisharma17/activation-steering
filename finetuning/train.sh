#!/bin/bash

#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --job-name="finetune baseline"
#SBATCH --mail-user=arushi17@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="logs/slurm-%j.out"


export HF_HOME=/lustre/hdd/LAS/jannesar-lab/arushi/
cd /lustre/hdd/LAS/jannesar-lab/arushi
source myenv/bin/activate
cd activation-steering-orig/finetuning/

python train_codellama_lora.py \
  --train_jsonl train.jsonl \
  --eval_jsonl  valid.jsonl \
  --model_name  meta-llama/CodeLlama-7b-Instruct-hf \
  --out_dir     outputs/codellama_apr_lora \
  --num_train_epochs 2 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --lr 2e-4 \
  --warmup_ratio 0.03 \
  --save_steps 500 \
  --logging_steps 50

