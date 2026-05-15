#!/bin/bash
#SBATCH --account=f2025.coms.5990.01
#SBATCH --partition=instruction 
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=256G
#SBATCH --job-name="apr_sweep"
#SBATCH --mail-user=moulica9@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="logs/sweep-%j.out"
#SBATCH --error="logs/sweep-%j.err"

set -e
set -x

# ============================================
# CONFIGURATION
# ============================================
MODEL_ID=${1:-"Qwen/Qwen2.5-Coder-14B-Instruct"}

# ============================================
# TIMING: Start total timer
# ============================================
SCRIPT_START_TIME=$(date +%s)
SCRIPT_START_DATETIME=$(date '+%Y-%m-%d %H:%M:%S')

# Fixed parameters
PAIRS_PATH="/work/classtmp/moulica9/activation-steering/defects4j_all_bugs.jsonl"
FEWSHOT_K=3
LIMIT=166
START=0

cd /work/classtmp/moulica9/activation-steering || exit 1
source myenv/bin/activate || exit 1
export HF_TOKEN=*****
export CUDA_VISIBLE_DEVICES=0

# ============================================
# CREATE MASTER RUN DIRECTORY
# ============================================
DATASET_NAME=$(basename "$PAIRS_PATH" .jsonl)
DATASET_NAME=$(echo "$DATASET_NAME" | tr '[:upper:]' '[:lower:]' | tr '-' '_' | tr '.' '_')

MODEL_SAFE=$(echo "$MODEL_ID" | tr '/' '_' | tr '[:upper:]' '[:lower:]' | tr '-' '_' | tr '.' '_')
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Master directory - everything goes here
RUN_DIR="deliverables/${DATASET_NAME}_${MODEL_SAFE}_${RUN_TIMESTAMP}"
mkdir -p "$RUN_DIR/logs"

echo "========================================="
echo "Master Run Directory: $RUN_DIR"
echo "========================================="

# Results CSV
RESULTS_CSV="${RUN_DIR}/results.csv"

# Vector path
VECTOR_PATH="${RUN_DIR}/steering_vector"

# ============================================
# SAVE RUN CONFIGURATION
# ============================================
cat > "${RUN_DIR}/run_config.txt" << EOF
========================================
HYPERPARAMETER SWEEP CONFIGURATION
========================================
Run ID: ${RUN_TIMESTAMP}
Start Time: ${SCRIPT_START_DATETIME}
Model: ${MODEL_ID}
Pairs Path: ${PAIRS_PATH}
Start Index: ${START}
Limit: ${LIMIT}
Few-shot K: ${FEWSHOT_K}
Vector Path: ${VECTOR_PATH}.svec
Results CSV: ${RESULTS_CSV}
========================================
EOF

# Create CSV header
echo "run_id,model,num_layers,layers,strength,fewshot_k,limit,baseline_correct,baseline_total,baseline_accuracy,baseline_invalid,steered_correct,steered_total,steered_accuracy,steered_invalid,improvement,run_time_seconds,timestamp" > "$RESULTS_CSV"

# ============================================
# MODEL-SPECIFIC CONFIGURATIONS
# ============================================
# Define exact layer counts and configurations for each model

declare -A MODEL_LAYERS
declare -A MODEL_CONFIGS

# CodeLlama models (32 layers: 0-31)
MODEL_LAYERS["codellama/CodeLlama-7b-hf"]=32
MODEL_LAYERS["codellama/CodeLlama-7b-Instruct-hf"]=32
MODEL_LAYERS["codellama/CodeLlama-13b-hf"]=40
MODEL_LAYERS["codellama/CodeLlama-13b-Instruct-hf"]=40
MODEL_LAYERS["codellama/CodeLlama-34b-hf"]=48
MODEL_LAYERS["codellama/CodeLlama-34b-Instruct-hf"]=48

# Qwen2.5-Coder models
MODEL_LAYERS["Qwen/Qwen2.5-Coder-7B"]=28
MODEL_LAYERS["Qwen/Qwen2.5-Coder-7B-Instruct"]=28
MODEL_LAYERS["Qwen/Qwen2.5-Coder-14B"]=48
MODEL_LAYERS["Qwen/Qwen2.5-Coder-14B-Instruct"]=48
MODEL_LAYERS["Qwen/Qwen2.5-Coder-32B"]=64
MODEL_LAYERS["Qwen/Qwen2.5-Coder-32B-Instruct"]=64

# Qwen (original) models
MODEL_LAYERS["Qwen/Qwen-7B"]=32
MODEL_LAYERS["Qwen/Qwen-7B-Chat"]=32
MODEL_LAYERS["Qwen/Qwen-14B"]=40
MODEL_LAYERS["Qwen/Qwen-14B-Chat"]=40

# DeepSeek-Coder models
MODEL_LAYERS["deepseek-ai/deepseek-coder-6.7b-base"]=32
MODEL_LAYERS["deepseek-ai/deepseek-coder-6.7b-instruct"]=32
MODEL_LAYERS["deepseek-ai/deepseek-coder-33b-base"]=48
MODEL_LAYERS["deepseek-ai/deepseek-coder-33b-instruct"]=48

# StarCoder models
MODEL_LAYERS["bigcode/starcoder"]=40
MODEL_LAYERS["bigcode/starcoder2-15b"]=40
MODEL_LAYERS["bigcode/starcoder2-7b"]=32

# WizardCoder models
MODEL_LAYERS["WizardLM/WizardCoder-15B-V1.0"]=40
MODEL_LAYERS["WizardLM/WizardCoder-Python-7B-V1.0"]=32

# ============================================
# GET LAYER COUNT FOR CURRENT MODEL
# ============================================
NUM_LAYERS=${MODEL_LAYERS[$MODEL_ID]}

if [ -z "$NUM_LAYERS" ]; then
    echo "========================================="
    echo "WARNING: Unknown model '$MODEL_ID'"
    echo "========================================="
    echo "Attempting to auto-detect number of layers..."
    
    # Try to detect layers programmatically
    NUM_LAYERS=$(python3 << 'PYEOF'
import sys
try:
    from transformers import AutoConfig
    import os
    
    model_id = os.environ.get('MODEL_ID')
    if not model_id:
        sys.exit(1)
    
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    
    # Try different attribute names that models use
    num_layers = None
    if hasattr(config, 'num_hidden_layers'):
        num_layers = config.num_hidden_layers
    elif hasattr(config, 'n_layer'):
        num_layers = config.n_layer
    elif hasattr(config, 'num_layers'):
        num_layers = config.num_layers
    
    if num_layers:
        print(num_layers)
    else:
        sys.exit(1)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
)
    
    if [ -z "$NUM_LAYERS" ] || [ "$NUM_LAYERS" -eq 0 ]; then
        echo "========================================="
        echo "ERROR: Could not determine number of layers"
        echo "========================================="
        echo ""
        echo "Please add this model to the MODEL_LAYERS mapping in the script:"
        echo "  MODEL_LAYERS[\"$MODEL_ID\"]=XX"
        echo ""
        echo "Or run this Python code to find the layer count:"
        echo "  from transformers import AutoConfig"
        echo "  config = AutoConfig.from_pretrained('$MODEL_ID')"
        echo "  print(config.num_hidden_layers)"
        echo ""
        exit 1
    fi
    
    echo "✓ Auto-detected $NUM_LAYERS layers"
fi

echo "========================================="
echo "Model: $MODEL_ID"
echo "Number of Layers: $NUM_LAYERS (0-$((NUM_LAYERS-1)))"
echo "========================================="

# ============================================
# GENERATE LAYER CONFIGURATIONS
# ============================================
# Generate layer configs based on actual number of layers
# Focus on middle to later layers (around 30-90% depth)

LAYER_CONFIGS=()

# Calculate key depths
START_LAYER=$((NUM_LAYERS * 30 / 100))  # 30% depth
END_LAYER=$((NUM_LAYERS - 1))            # Last layer

# Generate configurations with 5 consecutive layers each
# Spacing them out across the 30-100% range
RANGE=$((END_LAYER - START_LAYER))
NUM_CONFIGS=8  # Number of configurations to generate

for i in $(seq 0 $((NUM_CONFIGS - 1))); do
    # Calculate center layer for this config
    CENTER=$((START_LAYER + (RANGE * i / (NUM_CONFIGS - 1))))
    
    # Create 5-layer window around center
    L1=$((CENTER - 2))
    L2=$((CENTER - 1))
    L3=$CENTER
    L4=$((CENTER + 1))
    L5=$((CENTER + 2))
    
    # Clamp to valid range
    if [ $L1 -lt $START_LAYER ]; then
        L1=$START_LAYER
        L2=$((L1 + 1))
        L3=$((L1 + 2))
        L4=$((L1 + 3))
        L5=$((L1 + 4))
    fi
    
    if [ $L5 -gt $END_LAYER ]; then
        L5=$END_LAYER
        L4=$((L5 - 1))
        L3=$((L5 - 2))
        L2=$((L5 - 3))
        L1=$((L5 - 4))
    fi
    
    # Only add if all layers are within bounds
    if [ $L1 -ge 0 ] && [ $L5 -lt $NUM_LAYERS ]; then
        CONFIG="${L1},${L2},${L3},${L4},${L5}"
        LAYER_CONFIGS+=("$CONFIG")
    fi
done

# Remove duplicates (can happen with small models)
LAYER_CONFIGS=($(printf '%s\n' "${LAYER_CONFIGS[@]}" | sort -u))

echo ""
echo "Generated ${#LAYER_CONFIGS[@]} layer configurations:"
for i in "${!LAYER_CONFIGS[@]}"; do
    echo "  Config $((i+1)): ${LAYER_CONFIGS[$i]}"
done
echo ""

STRENGTHS=(1.0 1.5 2.0 2.5 3.0)

# Save hyperparameters
cat > "${RUN_DIR}/hyperparameters.txt" << EOF
========================================
HYPERPARAMETER SEARCH SPACE
========================================
Model: ${MODEL_ID}
Total Layers: ${NUM_LAYERS} (0-$((NUM_LAYERS-1)))

Layer Configurations (${#LAYER_CONFIGS[@]} configs):
EOF
for i in "${!LAYER_CONFIGS[@]}"; do
    echo "  Config $((i+1)): ${LAYER_CONFIGS[$i]}" >> "${RUN_DIR}/hyperparameters.txt"
done
cat >> "${RUN_DIR}/hyperparameters.txt" << EOF

Strength Values (${#STRENGTHS[@]} values):
  ${STRENGTHS[@]}

Total Experiments: $((${#LAYER_CONFIGS[@]} * ${#STRENGTHS[@]}))
========================================
EOF

RUN_ID=1

echo ""
echo "========================================="
echo "Starting Hyperparameter Sweep"
echo "========================================="
echo "Model: $MODEL_ID"
echo "Run Dir: $RUN_DIR"
echo "Start time: $SCRIPT_START_DATETIME"
echo "Total configurations: $((${#LAYER_CONFIGS[@]} * ${#STRENGTHS[@]}))"
echo "========================================="

# ============================================
# GENERATE STEERING VECTOR
# ============================================
if [ ! -f "${VECTOR_PATH}.svec" ]; then
    echo ""
    echo "Generating steering vector..."
    
    VECTOR_START=$(date +%s)
    python demo-extract.py \
        --model_id "$MODEL_ID" \
        --vector_name "$VECTOR_PATH" 2>&1 | tee "${RUN_DIR}/logs/vector_extraction.log"
    VECTOR_END=$(date +%s)
    VECTOR_TIME=$((VECTOR_END - VECTOR_START))
    
    if [ ! -f "${VECTOR_PATH}.svec" ]; then
        echo "ERROR: Failed to create steering vector"
        exit 1
    fi
    
    echo "✓ Steering vector created in ${VECTOR_TIME}s"
    echo "Vector extraction time: ${VECTOR_TIME}s" > "${RUN_DIR}/vector_info.txt"
    echo "Vector size: $(du -h "${VECTOR_PATH}.svec" | cut -f1)" >> "${RUN_DIR}/vector_info.txt"
else
    echo "✓ Reusing existing steering vector"
fi

echo ""
echo "Starting evaluation runs..."
echo ""

# ============================================
# RUN EVALUATIONS
# ============================================
for STRENGTH in "${STRENGTHS[@]}"; do
    for LAYERS in "${LAYER_CONFIGS[@]}"; do
        echo "========================================="
        echo "Run #${RUN_ID}: Layers=${LAYERS}, Strength=${STRENGTH}"
        echo "Started at: $(date)"
        echo "========================================="
        
        RUN_START_TIME=$(date +%s)
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        
        # Run evaluation - save log directly to master directory
        LAYERS_CLEAN=$(echo "$LAYERS" | sed 's/,/_/g')
        RUN_LOG="${RUN_DIR}/logs/run_${RUN_ID}_L${LAYERS_CLEAN}_S${STRENGTH}.log"
        
        if python ab_apr_eval.py \
            --pairs_path "$PAIRS_PATH" \
            --start $START \
            --limit $LIMIT \
            --fewshot_k $FEWSHOT_K \
            --model_id "$MODEL_ID" \
            --compare \
            --vector_path "$VECTOR_PATH" \
            --strength $STRENGTH \
            --layers "$LAYERS" \
            --show_n 1 2>&1 | tee "$RUN_LOG"; then
            
            RUN_END_TIME=$(date +%s)
            RUN_ELAPSED=$((RUN_END_TIME - RUN_START_TIME))
            
            echo "✓ Evaluation completed in ${RUN_ELAPSED}s"
            
            # Extract results
            BASELINE_MATCH=$(grep -oP '\[Baseline\] Accuracy: \K\d+/\d+ = [\d.]+%' "$RUN_LOG" | head -1)
            STEERED_MATCH=$(grep -oP '\[Steered \] Accuracy: \K\d+/\d+ = [\d.]+%' "$RUN_LOG" | head -1)
            BASELINE_INVALID=$(grep -oP '\[Baseline\].*Invalid: \K\d+/\d+' "$RUN_LOG" | head -1)
            STEERED_INVALID=$(grep -oP '\[Steered \].*Invalid: \K\d+/\d+' "$RUN_LOG" | head -1)
            
            if [ -n "$BASELINE_MATCH" ] && [ -n "$STEERED_MATCH" ]; then
                BASELINE_CORRECT=$(echo "$BASELINE_MATCH" | cut -d'/' -f1)
                BASELINE_TOTAL=$(echo "$BASELINE_MATCH" | cut -d'/' -f2 | cut -d'=' -f1 | xargs)
                BASELINE_ACC=$(echo "$BASELINE_MATCH" | grep -oP '[\d.]+%' | tr -d '%')
                
                STEERED_CORRECT=$(echo "$STEERED_MATCH" | cut -d'/' -f1)
                STEERED_TOTAL=$(echo "$STEERED_MATCH" | cut -d'/' -f2 | cut -d'=' -f1 | xargs)
                STEERED_ACC=$(echo "$STEERED_MATCH" | grep -oP '[\d.]+%' | tr -d '%')
                
                BASELINE_INV=$(echo "$BASELINE_INVALID" | cut -d'/' -f1)
                STEERED_INV=$(echo "$STEERED_INVALID" | cut -d'/' -f1)
                
                IMPROVEMENT=$(awk "BEGIN {printf \"%.4f\", $STEERED_ACC - $BASELINE_ACC}")
                
                # Save to CSV
                echo "${RUN_ID},${MODEL_ID},${NUM_LAYERS},\"${LAYERS}\",${STRENGTH},${FEWSHOT_K},${LIMIT},${BASELINE_CORRECT},${BASELINE_TOTAL},${BASELINE_ACC},${BASELINE_INV},${STEERED_CORRECT},${STEERED_TOTAL},${STEERED_ACC},${STEERED_INV},${IMPROVEMENT},${RUN_ELAPSED},${TIMESTAMP}" >> "$RESULTS_CSV"
                
                echo "✓ Results: Baseline=${BASELINE_ACC}%, Steered=${STEERED_ACC}%, Improvement=${IMPROVEMENT}%, Time=${RUN_ELAPSED}s"
            else
                RUN_END_TIME=$(date +%s)
                RUN_ELAPSED=$((RUN_END_TIME - RUN_START_TIME))
                
                echo "✗ Failed to extract results"
                echo "${RUN_ID},${MODEL_ID},${NUM_LAYERS},\"${LAYERS}\",${STRENGTH},${FEWSHOT_K},${LIMIT},ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,${RUN_ELAPSED},${TIMESTAMP}" >> "$RESULTS_CSV"
            fi
        else
            RUN_END_TIME=$(date +%s)
            RUN_ELAPSED=$((RUN_END_TIME - RUN_START_TIME))
            
            echo "✗ Evaluation failed (runtime: ${RUN_ELAPSED}s)"
            echo "${RUN_ID},${MODEL_ID},${NUM_LAYERS},\"${LAYERS}\",${STRENGTH},${FEWSHOT_K},${LIMIT},FAILED,FAILED,FAILED,FAILED,FAILED,FAILED,FAILED,FAILED,FAILED,${RUN_ELAPSED},${TIMESTAMP}" >> "$RESULTS_CSV"
        fi
        
        RUN_ID=$((RUN_ID + 1))
        echo "Completed at: $(date)"
        echo ""
        
        sleep 2
    done
done

# ============================================
# FINAL SUMMARY
# ============================================
SCRIPT_END_TIME=$(date +%s)
SCRIPT_END_DATETIME=$(date '+%Y-%m-%d %H:%M:%S')
TOTAL_ELAPSED=$((SCRIPT_END_TIME - SCRIPT_START_TIME))

HOURS=$((TOTAL_ELAPSED / 3600))
MINUTES=$(((TOTAL_ELAPSED % 3600) / 60))
SECONDS=$((TOTAL_ELAPSED % 60))

# Create final report
cat > "${RUN_DIR}/final_report.txt" << EOF
========================================
HYPERPARAMETER SWEEP FINAL REPORT
========================================
Model: ${MODEL_ID}
Model Layers: ${NUM_LAYERS}
Run ID: ${RUN_TIMESTAMP}
Start Time: ${SCRIPT_START_DATETIME}
End Time: ${SCRIPT_END_DATETIME}
Total Runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s (${TOTAL_ELAPSED}s)
Total Runs: $((RUN_ID - 1))

RESULTS ANALYSIS:
========================================
EOF

# Generate statistics
tail -n +2 "$RESULTS_CSV" | python3 -c "
import sys
import csv

sum_imp = 0
count = 0
max_imp = -999
max_config = ''
max_layers = ''
max_strength = ''
total_time = 0
min_time = 999999
max_time = 0
failed_count = 0

reader = csv.DictReader(sys.stdin)
for row in reader:
    if row['improvement'] == 'ERROR' or row['improvement'] == 'FAILED':
        failed_count += 1
        continue
    
    try:
        imp = float(row['improvement'])
        runtime = int(row['run_time_seconds'])
        
        sum_imp += imp
        count += 1
        total_time += runtime
        
        if runtime < min_time:
            min_time = runtime
        if runtime > max_time:
            max_time = runtime
        
        if imp > max_imp:
            max_imp = imp
            max_layers = row['layers']
            max_strength = row['strength']
    except (ValueError, KeyError):
        failed_count += 1

print(f'Successful Runs: {count}')
print(f'Failed Runs: {failed_count}')
print(f'')
print(f'PERFORMANCE METRICS:')
print(f'  Average Improvement: {sum_imp/count:.4f}%' if count > 0 else '  Average Improvement: N/A')
print(f'  Best Improvement: {max_imp:.4f}%' if count > 0 else '  Best Improvement: N/A')
print(f'  Best Configuration:')
print(f'    Layers: {max_layers}')
print(f'    Strength: {max_strength}')
print(f'')
print(f'TIMING STATISTICS:')
print(f'  Average Run Time: {total_time/count:.1f}s' if count > 0 else '  Average Run Time: N/A')
print(f'  Fastest Run: {min_time}s' if min_time < 999999 else '  Fastest Run: N/A')
print(f'  Slowest Run: {max_time}s' if max_time > 0 else '  Slowest Run: N/A')
print(f'  Total Evaluation Time: {total_time/3600:.2f}h ({total_time}s)' if count > 0 else '  Total Evaluation Time: N/A')
" >> "${RUN_DIR}/final_report.txt"

echo "" >> "${RUN_DIR}/final_report.txt"
echo "========================================" >> "${RUN_DIR}/final_report.txt"

# Print summary
cat "${RUN_DIR}/final_report.txt"

echo ""
echo "========================================="
echo "✅ SWEEP COMPLETE"
echo "========================================="
echo "📦 Output Directory: ${RUN_DIR}/"
echo "📊 Results CSV: ${RUN_DIR}/results.csv"
echo "📄 Final Report: ${RUN_DIR}/final_report.txt"
echo "🧠 Steering Vector: ${RUN_DIR}/steering_vector.svec"
echo ""
echo "Directory contents:"
ls -lh "${RUN_DIR}/"
echo ""
echo "To package:"
echo "  tar -czf ${MODEL_SAFE}_${RUN_TIMESTAMP}.tar.gz -C deliverables/ ${MODEL_SAFE}_${RUN_TIMESTAMP}/"
echo "========================================="
