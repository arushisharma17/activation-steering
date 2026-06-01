#M_PRJ_ROOT dir: has BugsInPy and singulairty for BugsInPy to run.
export M_PRJ_ROOT=/lustre/hdd/LAS/jannesar-lab/raoki/bugsinpy/0.0.1

export M_APP_SIF=/lustre/hdd/LAS/jannesar-lab/raoki/bugsinpy/0.0.1/BugsInPy.sif
export APPTAINERENV_PATH='/lustre/hdd/LAS/jannesar-lab/raoki/bugsinpy/0.0.1/BugsInPy/framework/bin:$PATH'

module purge
module load apptainer

#create alias for all commands in ./BugsInPy/framework/bin/
alias bugsinpy_shell='apptainer shell --cleanenv $M_APP_SIF "$@". '

