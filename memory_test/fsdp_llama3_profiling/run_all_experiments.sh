#!/bin/bash
# Run all FSDP Llama3 memory profiling experiments in isolated Python processes
# This prevents memory pool contamination between experiments

echo "========================================================================"
echo "FSDP Llama3 Memory Profiling Experiment Suite"
echo "Each experiment runs in a fresh Python process to avoid contamination"
echo "========================================================================"
echo ""

# Check for HuggingFace token
if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: HF_TOKEN environment variable is not set"
    echo "Please set it with: export HF_TOKEN=your_token_here"
    return 1
fi

# Configuration
# MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME="meta-llama/Llama-3.2-1B"
BATCH_SIZE=2
NUM_ITER=1
WARMUP_ITER=1
VOCAB_SIZE=128256
# VOCAB_SIZE=32000
LEARNING_RATE=1e-5
SIGMA=1.0
MAX_GRAD_NORM=1.0

# Accuracy test mode (set to "true" to enable)
# When enabled: uses low noise (sigma=0.01, epsilon≈1e-3) and high clip norm (10000)
# to verify algorithm consistency
ACCURACY_TEST_MODE=false
ACCURACY_TEST_SIGMA=0.01
ACCURACY_TEST_MAX_GRAD_NORM=10000
ACCURACY_TEST_RANDOM_SEED=42

# Sequence lengths to test
SEQ_LENGTHS=(1024)

# Modes to test
# Available FSDP modes: no_dp, ghost_fsdp, flash_fsdp, flash_fsdp_bk, ghost_fsdp_bk, flash_fsdp_fuse, flash_fsdp_fuse_bk
# Available Single-GPU modes: no_dp_single, ghost, flash, flash_bk, ghost_bk
# MODES=("no_dp" "ghost_fsdp" "flash_fsdp" "flash_fsdp_fuse")
MODES=("flash_fsdp_fuse" "flash_fsdp_fuse_bk" "flash_fsdp_bk" "flash_fsdp" "no_dp") 
# MODES=("flash_fsdp_fuse_bk") 

# Output directory
OUTPUT_DIR="results"
mkdir -p "$OUTPUT_DIR"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$OUTPUT_DIR/run_$TIMESTAMP"
mkdir -p "$RUN_DIR"

echo "Output directory: $RUN_DIR"
echo "Model: $MODEL_NAME"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Sequence lengths: ${SEQ_LENGTHS[@]}"
echo "Modes: ${MODES[@]}"

if [ "$ACCURACY_TEST_MODE" = "true" ]; then
    echo ""
    echo "🔬 ACCURACY TEST MODE ENABLED"
    echo "   Sigma: $ACCURACY_TEST_SIGMA (epsilon ≈ 1e-3)"
    echo "   Max Grad Norm: $ACCURACY_TEST_MAX_GRAD_NORM"
    echo "   Random Seed: $ACCURACY_TEST_RANDOM_SEED"
fi

echo ""

# Function to run a single experiment
run_experiment() {
    local mode=$1
    local seq_len=$2
    local output_file="$RUN_DIR/${mode}_seq${seq_len}_result.json"
    
    echo "========================================================================"
    echo "Running: mode=$mode, seq_length=$seq_len"
    echo "Output: $output_file"
    echo "========================================================================"
    
    # Activate virtual environment if needed
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi
    
    # Determine parameters based on accuracy test mode
    if [ "$ACCURACY_TEST_MODE" = "true" ]; then
        CURRENT_SIGMA=$ACCURACY_TEST_SIGMA
        CURRENT_MAX_GRAD_NORM=$ACCURACY_TEST_MAX_GRAD_NORM
        ACCURACY_FLAG="--accuracy-test --random-seed $ACCURACY_TEST_RANDOM_SEED"
    else
        CURRENT_SIGMA=$SIGMA
        CURRENT_MAX_GRAD_NORM=$MAX_GRAD_NORM
        ACCURACY_FLAG=""
    fi
    
    # Run experiment in isolated process
    python single_experiment.py \
        --mode "$mode" \
        --seq-length $seq_len \
        --batch-size $BATCH_SIZE \
        --num-iter $NUM_ITER \
        --warmup-iter $WARMUP_ITER \
        --output "$output_file" \
        --model-name "$MODEL_NAME" \
        --token "$HF_TOKEN" \
        --vocab-size $VOCAB_SIZE \
        --learning-rate $LEARNING_RATE \
        --sigma $CURRENT_SIGMA \
        --max-grad-norm $CURRENT_MAX_GRAD_NORM \
        $ACCURACY_FLAG
    
    local exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        echo "❌ Experiment failed with exit code $exit_code"
        echo "Continuing with remaining experiments..."
    else
        echo "✓ Completed: mode=$mode, seq_length=$seq_len"
    fi
    
    # Wait between experiments to ensure full cleanup
    echo "Waiting for cleanup..."
    sleep 5
    
    echo ""
}

# Run all experiments
echo "Starting experiment sequence..."
echo ""

total_experiments=$((${#MODES[@]} * ${#SEQ_LENGTHS[@]}))
current_experiment=0

for seq_len in "${SEQ_LENGTHS[@]}"; do
    for mode in "${MODES[@]}"; do
        current_experiment=$((current_experiment + 1))
        echo ""
        echo "════════════════════════════════════════════════════════════════════"
        echo "Progress: Experiment $current_experiment / $total_experiments"
        echo "════════════════════════════════════════════════════════════════════"
        run_experiment "$mode" "$seq_len"
    done
done

echo "========================================================================"
echo "All experiments completed!"
echo "========================================================================"
echo ""

# Generate visualizations
echo "Generating visualizations..."
if [ -f "visualize_results.py" ]; then
    python visualize_results.py \
        --input-dir "$RUN_DIR" \
        --output-dir "$RUN_DIR/visualizations"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Visualizations generated successfully!"
        echo "✅ Visualizations saved to: $RUN_DIR/visualizations"
    else
        echo "⚠️  Visualization script failed"
    fi
else
    echo "⚠️  visualize_results.py not found, skipping visualization"
fi

echo ""
echo "========================================================================"
echo "SUMMARY"
echo "========================================================================"

# Print summary table header
printf "%-15s %-12s %15s %15s\n" "Mode" "Seq Length" "Peak Mem (GB)" "Avg Time (ms)"
echo "------------------------------------------------------------------------"

# Print results for each experiment
for seq_len in "${SEQ_LENGTHS[@]}"; do
    for mode in "${MODES[@]}"; do
        result_file="$RUN_DIR/${mode}_seq${seq_len}_result.json"
        if [ -f "$result_file" ]; then
            peak_mem=$(python -c "import json; data=json.load(open('$result_file')); print(f\"{data['peak_memory_gb']:.2f}\")" 2>/dev/null || echo "N/A")
            avg_time=$(python -c "import json; data=json.load(open('$result_file')); print(f\"{data['avg_time_ms']:.2f}\")" 2>/dev/null || echo "N/A")
            printf "%-15s %-12s %15s %15s\n" "$mode" "$seq_len" "$peak_mem" "$avg_time"
        else
            printf "%-15s %-12s %15s %15s\n" "$mode" "$seq_len" "FAILED" "FAILED"
        fi
    done
    echo "------------------------------------------------------------------------"
done

echo ""
echo "✅ Complete pipeline finished!"
echo "✅ Results directory: $RUN_DIR"
if [ -d "$RUN_DIR/visualizations" ]; then
    echo "✅ Visualizations: $RUN_DIR/visualizations"
fi
echo ""

