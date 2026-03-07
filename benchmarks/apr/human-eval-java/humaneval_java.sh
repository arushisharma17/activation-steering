#!/bin/bash
#SBATCH --job-name=humaneval-java
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --output=humaneval-java-%j.out

#########################
# 0. Modules
#########################
module purge
module load openjdk/17.0.11_9-w2j2eju
module load maven/3.8.4-ieg7dba   # adjust if you prefer the other 3.8.4

echo "Node: $(hostname)"
echo "Working directory at submit time: $SLURM_SUBMIT_DIR"

echo "Java version:"
java -version
echo
echo "Maven version:"
mvn -version
echo

#########################
# 1. Maven repo on Lustre
#########################
export MAVEN_REPO="/lustre/hdd/LAS/jannesar-lab/arushi/maven-repo"
echo "Using Maven local repo: $MAVEN_REPO"

mkdir -p "$MAVEN_REPO"
chmod -R u+rwX "$MAVEN_REPO"

#########################
# 2. Work directory for HumanEval-Java
#########################
WORKDIR="/lustre/hdd/LAS/jannesar-lab/arushi/"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "Job work directory: $(pwd)"
echo

#########################
# 3. Clone or update HumanEval-Java
#########################
REPO_DIR="$WORKDIR/human-eval-java"

if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning HumanEval-Java repo..."
    git clone https://github.com/ASSERT-KTH/human-eval-java.git
else
    echo "Repo already exists, updating..."
    cd "$REPO_DIR"
    git pull --rebase
    cd "$WORKDIR"
fi

cd "$REPO_DIR"

echo "Now in repo: $(pwd)"
echo

#########################
# 4. Run tests (all or specific)
#########################
# If TEST_NAME is set (e.g., TEST_HumanEval_0), run only that test class.
# Otherwise, run the full test suite.
#
# You can set TEST_NAME by:
#   sbatch --export=TEST_NAME=TEST_HumanEval_0 humaneval_java.slurm
#
# Or put it in the header:
#   #SBATCH --export=TEST_NAME=TEST_HumanEval_0

if [ -n "$TEST_NAME" ]; then
    echo "Running Maven tests for specific class: $TEST_NAME"
    mvn -Dmaven.repo.local="$MAVEN_REPO" test -Dtest="$TEST_NAME"
else
    echo "Running ALL Maven tests for HumanEval-Java..."
    mvn -Dmaven.repo.local="$MAVEN_REPO" test
fi

echo
echo "=== HumanEval-Java test run complete ==="

