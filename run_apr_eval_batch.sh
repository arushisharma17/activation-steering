#!/bin/bash
#SBATCH --account=f2025.coms.5990.01
#SBATCH --partition=instruction 
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --job-name="apr_overnight"
#SBATCH --mail-user=moulica9@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="logs/slurm-%j.out"
#SBATCH --error="logs/slurm-%j.err"

set -e

cd /work/classtmp/moulica9/activation-steering
source myenv/bin/activate

export HF_TOKEN=${HF_TOKEN:-$(cat ~/.hf_token 2>/dev/null)}
export CUDA_VISIBLE_DEVICES=0

mkdir -p results logs

RESULTS_FILE="results/apr_overnight_sweep_$(date +%Y%m%d_%H%M%S).csv"
echo "run_id,layers,strength,fewshot_k,start,limit,baseline_accuracy,baseline_correct,baseline_total,baseline_invalid,steered_accuracy,steered_correct,steered_total,steered_invalid,improvement" > $RESULTS_FILE

echo "Results will be saved to: $RESULTS_FILE"
echo ""

echo "Generating steering vector..."
python demo-extract.py

# ============================================
# OVERNIGHT CONFIGURATION (72 runs, ~2.5 hrs)
# Optimized for 129 data pairs
# ============================================

declare -a LAYER_CONFIGS=(
    "25,26,27"                   # Last 3 layers
    "23,24,25,26,27"            # Last 5 layers
    "22,23,24,25,26,27"         # Last 6 layers
    "21,22,23,24,25,26,27"      # Last 7 layers
)

declare -a STRENGTHS=(
    "2.0"
    "2.5"
    "3.0"
)

declare -a FEWSHOT_K_VALUES=(
    "3"
    "5"
    "7"
)

# Using your full dataset (129 pairs)
declare -a DATA_RANGES=(
    "0:100"      # 100 examples
    "0:129"      # ALL data (full dataset)
)

TOTAL_RUNS=$((${#LAYER_CONFIGS[@]} * ${#STRENGTHS[@]} * ${#FEWSHOT_K_VALUES[@]} * ${#DATA_RANGES[@]}))

echo "========================================"
echo "OVERNIGHT SWEEP - Full Dataset"
echo "Dataset size: 129 pairs"
echo "Layer configs: ${#LAYER_CONFIGS[@]}"
echo "Strengths: ${#STRENGTHS[@]}"
echo "Few-shot K values: ${#FEWSHOT_K_VALUES[@]}"
echo "Data ranges: ${#DATA_RANGES[@]}"
echo "Total runs: $TOTAL_RUNS"
echo "Estimated time: ~$((TOTAL_RUNS * 2)) minutes"
echo "Started: $(date)"
echo "========================================"
echo ""

RUN_ID=1
START_TIME=$(date +%s)

for LAYERS in "${LAYER_CONFIGS[@]}"; do
    for STRENGTH in "${STRENGTHS[@]}"; do
        for FEWSHOT_K in "${FEWSHOT_K_VALUES[@]}"; do
            for DATA_RANGE in "${DATA_RANGES[@]}"; do
                START=$(echo $DATA_RANGE | cut -d':' -f1)
                LIMIT=$(echo $DATA_RANGE | cut -d':' -f2)
                
                RUN_START=$(date +%s)
                
                echo "========================================"
                echo "Run $RUN_ID/$TOTAL_RUNS"
                echo "Layers: $LAYERS | Strength: $STRENGTH | K: $FEWSHOT_K | Range: $START:$LIMIT"
                echo "Started: $(date)"
                echo "========================================"
                
                python ab_apr_eval.py \
                    --pairs_path /work/classtmp/moulica9/activation-steering/hej_jsonl/pairs_compat.jsonl \
                    --start $START --limit $LIMIT \
                    --fewshot_k $FEWSHOT_K \
                    --model_id Qwen/Qwen2.5-Coder-7B-Instruct \
                    --compare \
                    --vector_path refusal_behavior_vector \
                    --strength $STRENGTH \
                    --layers $LAYERS \
                    --output_csv $RESULTS_FILE \
                    --run_id $RUN_ID \
                    --show_n 1
                
                EXIT_CODE=$?
                if [ $EXIT_CODE -ne 0 ]; then
                    echo "WARNING: Run $RUN_ID failed with exit code $EXIT_CODE"
                fi
                
                RUN_END=$(date +%s)
                RUN_DURATION=$((RUN_END - RUN_START))
                
                echo "Run $RUN_ID completed in ${RUN_DURATION}s"
                echo ""
                
                RUN_ID=$((RUN_ID + 1))
                
                # Progress update every 12 runs
                if [ $((RUN_ID % 12)) -eq 0 ]; then
                    CURRENT_TIME=$(date +%s)
                    ELAPSED=$((CURRENT_TIME - START_TIME))
                    AVG_TIME=$((ELAPSED / (RUN_ID - 1)))
                    REMAINING=$((TOTAL_RUNS - RUN_ID + 1))
                    ETA=$((REMAINING * AVG_TIME))
                    echo "========== PROGRESS UPDATE =========="
                    echo "Completed: $((RUN_ID-1))/$TOTAL_RUNS"
                    echo "Elapsed: $((ELAPSED/60))m"
                    echo "ETA: $((ETA/60))m remaining"
                    echo "====================================="
                    echo ""
                fi
                
                sleep 1
            done
        done
    done
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo "========================================"
echo "OVERNIGHT SWEEP COMPLETE!"
echo "========================================"
echo "Total runs: $TOTAL_RUNS"
echo "Total time: $((TOTAL_DURATION / 3600))h $((TOTAL_DURATION % 3600 / 60))m"
echo "Avg per run: $((TOTAL_DURATION / TOTAL_RUNS))s"
echo "Results saved to: $RESULTS_FILE"
echo "Completed at: $(date)"
echo "========================================"
echo ""

# Analysis
echo "TOP 10 CONFIGURATIONS:"
echo "----------------------"
tail -n +2 $RESULTS_FILE | sort -t, -k15 -rn | head -10 | \
    awk -F, 'BEGIN {
        printf "%-4s %-25s %-8s %-4s %-8s %-9s %-9s %-10s\n",
        "Run", "Layers", "Strength", "K", "Limit", "Baseline", "Steered", "Improve"
        print "--------------------------------------------------------------------------------"
    }
    {
        printf "%-4s %-25s %-8s %-4s %-8s %-9.2f %-9.2f %-10.4f\n",
        $1, $2, $3, $4, $6, $7*100, $11*100, $15
    }'

echo ""
echo "BEST OVERALL:"
tail -n +2 $RESULTS_FILE | sort -t, -k15 -rn | head -1 | \
    awk -F, '{
        printf "Run %s: Layers=%s, Strength=%s, K=%s, Limit=%s\n", $1, $2, $3, $4, $6
        printf "Baseline: %.2f%% → Steered: %.2f%% | Improvement: +%.4f\n", $7*100, $11*100, $15
    }'

echo ""
echo "ANALYSIS BY FEW-SHOT K:"
echo "-----------------------"
for k in 3 5 7; do
    tail -n +2 $RESULTS_FILE | awk -F, -v k="$k" '
        $4==k {
            sum_baseline += $7
            sum_steered += $11
            sum_improve += $15
            count++
        }
        END {
            if (count > 0) {
                printf "K=%d: Avg Baseline=%.2f%%, Avg Steered=%.2f%%, Avg Improvement=%+.4f (%d runs)\n",
                k, (sum_baseline/count)*100, (sum_steered/count)*100, sum_improve/count, count
            }
        }
    '
done

echo ""
echo "ANALYSIS BY DATA SIZE:"
echo "----------------------"
for limit in 100 129; do
    tail -n +2 $RESULTS_FILE | awk -F, -v lim="$limit" '
        $6==lim {
            sum_baseline += $7
            sum_steered += $11
            sum_improve += $15
            count++
        }
        END {
            if (count > 0) {
                printf "Limit=%d: Avg Baseline=%.2f%%, Avg Steered=%.2f%%, Avg Improvement=%+.4f (%d runs)\n",
                lim, (sum_baseline/count)*100, (sum_steered/count)*100, sum_improve/count, count
            }
        }
    '
done
