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


export HF_HOME=/lustre/hdd/LAS/jannesar-lab/arushi/
cd /lustre/hdd/LAS/jannesar-lab/arushi
source myenv/bin/activate
cd activation-steering-orig/docs/demo-data

#To create contrastive pairs
python apr_data.py /lustre/hdd/LAS/jannesar-lab/arushi/tssb_data_3M/filtered-0.jsonl --out behavior_refusal-apr.json --num_samples 2000

#To create apr questions
python make_apr_questions.py  /lustre/hdd/LAS/jannesar-lab/arushi/docs/demo-data/filtered-0.jsonl --out apr_questions_2k_5k.json --start 2000 --end 5000
