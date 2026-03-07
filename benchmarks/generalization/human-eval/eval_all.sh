#!/bin/bash
#
# Usage:
#   sbatch run_all.sh --baseline
#   sbatch run_all.sh --steered
#   sbatch run_all.sh         # default = both
#

#SBATCH --time=2-23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --gres=gpu:a100
#SBATCH --mem=128G
#SBATCH --job-name="humaneval-all"
#SBATCH --mail-user=arushi17@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="logs/slurm-%j.out"

export XDG_CACHE_HOME=/lustre/hdd/LAS/jannesar-lab/arushi/cache
export TRITON_CACHE_DIR=/lustre/hdd/LAS/jannesar-lab/arushi/cache/triton
export HF_HOME=/lustre/hdd/LAS/jannesar-lab/arushi/

cd /lustre/hdd/LAS/jannesar-lab/arushi/activation-steering/benchmarks/generalization/human-eval
mkdir -p logs


export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

nvidia-smi


# After all baseline + steered runs finish
# (pick an env where human_eval + pandas are installed)
source /lustre/hdd/LAS/jannesar-lab/arushi/.venv/bin/activate   # or .venv, whichever has human_eval
python eval_all.py
deactivate


echo "[INFO] All done."

