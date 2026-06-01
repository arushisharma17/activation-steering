#!/bin/bash
#SBATCH --job-name=bugsinpy_tests
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/lustre/hdd/LAS/jannesar-lab/raoki/bugsinpy/0.0.1/test_results_%j.log

BASE=/lustre/hdd/LAS/jannesar-lab/raoki/bugsinpy/0.0.1

module purge
module load apptainer

export CONDA_ENVS_PATH=$BASE/conda/envs
export CONDA_PKGS_DIRS=$BASE/conda/pkgs
export APPTAINERENV_CONDA_ENVS_PATH=$BASE/conda/envs
export APPTAINERENV_CONDA_PKGS_DIRS=$BASE/conda/pkgs

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"

apptainer exec --cleanenv \
  -B $BASE:$BASE \
  $BASE/BugsInPy.sif \
  bash -c "
    export PATH=$BASE/BugsInPy/framework/bin:\$PATH && \
    conda config --add envs_dirs $BASE/conda/envs && \
    bash $BASE/run_all_tests.sh
  "

echo "Job finished: $(date)"

