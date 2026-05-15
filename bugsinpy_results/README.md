#### 1. Create a workspace for the projects

Files: 
--`singularity.def`:  definition file to build singularity.def BugsInPy.sif. miniconda placed

--`BugsInPy.sif`: singularity built from singularity.def. Copy this container file to your workspace.

--`setup.sh`: make a few short cuts for commands, so that you do not need to type in the long commands when run them.  
How to use:  `$  .  ./setup.sh`

Env setup in your /<TO/YOUR/PATH>/bugsinpy/0.0.1/
#1.  need BugsInPy, BugsInPy.sif, and setup.sh

Env setup in your /<TO/YOUR/PATH>/bugsinpy/0.0.1/
#1.  need BugsInPy, BugsInPy.sif, and setup.sh

#### 2. Replace <TO/YOUR/PATH> with the actual file path 
your path:  `/<TO/YOUR/PATH>/bugsinpy/0.0.1`

    $ mkdir -p /<TO/YOUR/PATH>/bugsinpy/0.0.1
    $ export M_PRJ_ROOT=/<TO/YOUR/PATH>/bugsinpy/0.0.1
    $ cd $M_PRJ_ROOT

#### Sanity check 
`$ cat singularity.def`

Download the BugsInPy here
    
    $ git clone https://github.com/reproducing-research-projects/BugsInPy

Edit setup.sh to replace /ptmp/persistent/demo/bugsinpy/0.0.1 with your project root M_PRJ_ROOT.
So all alias will work in your setup.

    $ vi setup.sh.  


Bugsinpy does need user to create conda env during testing.
So let us run this inside of the container for easily handling conda issue.

#### 3. Shell-in the container.
Modify setup.sh by removing all bugsinpy commands, and adding the following line:
`alias bugsinpy_shell='apptainer shell --cleanenv $M_APP_SIF "$@". '`

Then source it
    
    $ .  ./setup.sh
    $ bugsinpy_shell

### >>>>>>Now inside of the container>>>>>>

    Apptrainer > mkdir -p my_project conda/envs/
    Apptainer > conda config --add envs_dirs /<TO/YOUR/PATH>/bugsinpy/0.0.1/conda/env

#### Example with fastapi: use py 3.8.3

    Apptainer >bugsinpy-checkout -p fastapi  -v 0 -i 2 -w /<TO/YOUR/PATH>/bugsinpy/0.0.1/my_project
    Apptainer> cd /<TO/YOUR/PATH>/bugsinpy/0.0.1/my_project/fastapi
    Apptainer> bugsinpy-compile
    
    #Please use 'conda create -n 6d77396cd44c7df1384dad69b48c4335 -y python=3.8.3' to
    Appatiner> conda create -n 6d77396cd44c7df1384dad69b48c4335 -y python=3.8.3

    #install the dependent packages by bugsinpy-compile :
    Apptainer> bugsinpy-compile

Run one more time  bugsinpy-compile to check all installed packages to satisfy the requirements.
You will see ~2 errors about the meta-generation. It is just warninga. All other packages are OK.

    Apptainer> bugsinpy-compile
    Apptainer> bugsinpy-test



## Run the script that automatically extracts the ground truth (buggy file + lines) for all bugs in the dataset

#### Upload to Nova

    scp extract_ground_truth.sh <NETID>@nova.its.iastate.edu:/<TO/YOUR/PATH>/bugsinpy/0.0.1/

#### Enter the container first

    . ./setup.sh
    bugsinpy_shell

#### Inside Apptainer shell:

    conda config --add envs_dirs /<TO/YOUR/PATH>/bugsinpy/0.0.1/conda/envs 
    bash /<TO/YOUR/PATH>/bugsinpy/0.0.1/extract_ground_truth.sh


### Run the script with SLURM job
Exit the Apptainer shell first.
    
    exit

Then run the script.

    cat > run_ground_truth.sh << 'EOF'
    #!/bin/bash
    #SBATCH --job-name=bugsinpy_gt
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=4
    #SBATCH --mem=16G
    #SBATCH --time=12:00:00
    #SBATCH --output=ground_truth_%j.log

    BASE=/<TO/YOUR/PATH>/bugsinpy/0.0.1

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
        conda config --add envs_dirs $BASE/conda/envs && \
        export PATH=$BASE/BugsInPy/framework/bin:\$PATH && \
        bash $BASE/extract_ground_truth.sh
    "

    echo "Job finished: $(date)"
    EOF

    sbatch run_ground_truth.sh
    squeue -u <USER_NAME>


Check job status

    squeue -u <USER_NAME>

Watch the log live as it runs

    tail -f ground_truth_*.log
