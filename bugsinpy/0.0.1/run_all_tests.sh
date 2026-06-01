#!/bin/bash
# =============================================================================
# run_all_tests.sh
# For each bug:
#   1. Checkout the project (buggy or fixed)
#   2. Replace the relevant files with our extracted buggy/fixed versions
#   3. Run bugsinpy-test and record pass/fail
#
# Output:
#   test_results.csv   - project, bug_id, version, result
#
# Runs INSIDE the Apptainer container via run_tests_slurm.sh
# =============================================================================

BASE=/lustre/hdd/LAS/jannesar-lab/raoki/bugsinpy/0.0.1
WORKSPACE=$BASE/tmp_test_checkout
PROJECTS_DIR=$BASE/BugsInPy/projects
BUGGY_DIR=$BASE/buggy_code
FIXED_DIR=$BASE/fixed_code
RESULTS_CSV=$BASE/test_results.csv

mkdir -p $WORKSPACE

# CSV header
echo "project,bug_id,version,result" > $RESULTS_CSV

echo "============================================="
echo " BugsInPy Test Runner"
echo " Started: $(date)"
echo "============================================="

# Parse test output and return pass/fail/error
# Priority: check FAILED/passed first (pytest output), then error
get_result() {
    local output="$1"
    # Check pytest summary line first - most reliable
    if echo "$output" | grep -qE "^FAILED |[0-9]+ failed"; then
        echo "fail"
    elif echo "$output" | grep -qE "[0-9]+ passed|passed in [0-9]"; then
        echo "pass"
    elif echo "$output" | grep -qE "^(OK|ok)$| ok in "; then
        echo "pass"
    elif echo "$output" | grep -qE "ERROR |[0-9]+ error"; then
        echo "error"
    else
        # Last resort - check exit code stored separately
        echo "unknown"
    fi
}

for project_path in $PROJECTS_DIR/*/; do
    project=$(basename $project_path)
    bugs_dir="$project_path/bugs"

    if [ ! -d "$bugs_dir" ]; then
        echo "[SKIP] No bugs dir for $project"
        continue
    fi

    for bug_path in $bugs_dir/*/; do
        bug_id=$(basename $bug_path)

        if ! [[ "$bug_id" =~ ^[0-9]+$ ]]; then
            continue
        fi

        BUG_BUGGY=$BUGGY_DIR/${project}_${bug_id}
        BUG_FIXED=$FIXED_DIR/${project}_${bug_id}

        if [ ! -d "$BUG_BUGGY" ] && [ ! -d "$BUG_FIXED" ]; then
            echo "[SKIP] No extracted files for $project #$bug_id"
            continue
        fi

        echo "---------------------------------------------"
        echo "Testing: $project bug #$bug_id  ($(date '+%H:%M:%S'))"

        PROJ_DIR=$WORKSPACE/$project

        # ----------------------------------------------------------------
        # Checkout and compile once to set up conda env
        # ----------------------------------------------------------------
        rm -rf $PROJ_DIR
        echo "  [1/5] Checking out base version..."
        bugsinpy-checkout -p $project -v 0 -i $bug_id -w $WORKSPACE 2>&1 | tail -2

        if [ ! -d "$PROJ_DIR" ]; then
            echo "  [ERROR] Checkout failed"
            echo "$project,$bug_id,buggy,error" >> $RESULTS_CSV
            echo "$project,$bug_id,fixed,error" >> $RESULTS_CSV
            continue
        fi

        cd $PROJ_DIR

        echo "  [2/5] Compiling..."
        compile_out=$(bugsinpy-compile 2>&1)
        if echo "$compile_out" | grep -q "conda create"; then
            conda_cmd=$(echo "$compile_out" | grep "conda create" | head -1 \
                | sed "s/Please use '//;s/' to.*//")
            echo "         Creating conda env: $conda_cmd"
            eval $conda_cmd 2>&1 | tail -2
            bugsinpy-compile 2>&1 | tail -2
            bugsinpy-compile 2>&1 | tail -2
        else
            bugsinpy-compile 2>&1 | tail -2
        fi

        # ----------------------------------------------------------------
        # Test BUGGY version
        # ----------------------------------------------------------------
        if [ -d "$BUG_BUGGY" ]; then
            echo "  [3/5] Replacing files with BUGGY version..."
            find $BUG_BUGGY -name "*.py" | while read extracted_file; do
                rel_path=${extracted_file#$BUG_BUGGY/}
                dest="$PROJ_DIR/$rel_path"
                mkdir -p "$(dirname $dest)"
                cp "$extracted_file" "$dest"
                echo "    replaced: $rel_path"
            done

            echo "  Running BUGGY test..."
            test_out=$(bugsinpy-test 2>&1)
            # Show just the pytest summary lines
            echo "$test_out" | grep -E "FAILED|passed|failed|error|ERROR|passed in|failed in" | tail -5

            buggy_result=$(get_result "$test_out")
            # Also check exit code
            bugsinpy-test > /dev/null 2>&1
            exit_code=$?
            if [ "$buggy_result" = "unknown" ]; then
                [ $exit_code -eq 0 ] && buggy_result="pass" || buggy_result="fail"
            fi

            echo "  Buggy result: $buggy_result"
            echo "$project,$bug_id,buggy,$buggy_result" >> $RESULTS_CSV
        else
            echo "$project,$bug_id,buggy,no_files" >> $RESULTS_CSV
        fi

        # ----------------------------------------------------------------
        # Test FIXED version
        # ----------------------------------------------------------------
        if [ -d "$BUG_FIXED" ]; then
            echo "  [4/5] Checking out FIXED version..."
            rm -rf $PROJ_DIR
            bugsinpy-checkout -p $project -v 1 -i $bug_id -w $WORKSPACE 2>&1 | tail -2

            if [ ! -d "$PROJ_DIR" ]; then
                echo "  [ERROR] Fixed checkout failed"
                echo "$project,$bug_id,fixed,error" >> $RESULTS_CSV
                cd $BASE
                continue
            fi

            cd $PROJ_DIR
            bugsinpy-compile 2>&1 | tail -2

            echo "  [5/5] Replacing files with FIXED version..."
            find $BUG_FIXED -name "*.py" | while read extracted_file; do
                rel_path=${extracted_file#$BUG_FIXED/}
                dest="$PROJ_DIR/$rel_path"
                mkdir -p "$(dirname $dest)"
                cp "$extracted_file" "$dest"
                echo "    replaced: $rel_path"
            done

            echo "  Running FIXED test..."
            test_out=$(bugsinpy-test 2>&1)
            echo "$test_out" | grep -E "FAILED|passed|failed|error|ERROR|passed in|failed in" | tail -5

            fixed_result=$(get_result "$test_out")
            bugsinpy-test > /dev/null 2>&1
            exit_code=$?
            if [ "$fixed_result" = "unknown" ]; then
                [ $exit_code -eq 0 ] && fixed_result="pass" || fixed_result="fail"
            fi

            echo "  Fixed result: $fixed_result"
            echo "$project,$bug_id,fixed,$fixed_result" >> $RESULTS_CSV
        else
            echo "$project,$bug_id,fixed,no_files" >> $RESULTS_CSV
        fi

        echo ""
        cd $BASE
    done
done

rm -rf $WORKSPACE

echo "============================================="
echo " DONE: $(date)"
echo " Results: $RESULTS_CSV"
echo "============================================="

echo ""
echo "Summary:"
total=$(tail -n +2 $RESULTS_CSV | wc -l)
pass=$(grep ",pass$" $RESULTS_CSV | wc -l)
fail=$(grep ",fail$" $RESULTS_CSV | wc -l)
error=$(grep ",error$" $RESULTS_CSV | wc -l)
unknown=$(grep ",unknown$" $RESULTS_CSV | wc -l)
echo "  Total   : $total"
echo "  Pass    : $pass"
echo "  Fail    : $fail"
echo "  Error   : $error"
echo "  Unknown : $unknown"
