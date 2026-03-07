#!/bin/bash
#SBATCH --job-name=java-maven-setup
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --output=java-maven-setup-%j.out

#########################
# 0. Modules
#########################
module purge
module load openjdk/17.0.11_9-w2j2eju
module load maven/3.8.4-ieg7dba   # or the other 3.8.4 version

echo "Node: $(hostname)"
echo "Java version:"
java -version
echo "Maven version:"
mvn -version

#########################
# 1. Create Maven repo on Lustre
#########################
export MAVEN_REPO="/lustre/hdd/LAS/jannesar-lab/arushi/maven-repo"

echo "Creating Maven repo at: $MAVEN_REPO"

mkdir -p "$MAVEN_REPO"
chmod -R u+rwX "$MAVEN_REPO"

#########################
# 2. Make a scratch directory for testing
#########################
WORKDIR="/lustre/hdd/LAS/jannesar-lab/arushi/maven-test-job"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "Working directory: $(pwd)"

#########################
# 3. Generate a sample Maven project
#########################
mvn -q \
  -Dmaven.repo.local="$MAVEN_REPO" \
  archetype:generate \
  -DgroupId=test.hpc \
  -DartifactId=maven-test \
  -DarchetypeArtifactId=maven-archetype-quickstart \
  -DarchetypeVersion=1.4 \
  -DinteractiveMode=false

#########################
# 4. Run tests
#########################
cd maven-test

echo "Running mvn test..."
mvn -q -Dmaven.repo.local="$MAVEN_REPO" test

echo "=== DONE ==="

