#!/bin/bash
# Run all Linear layer memory profiling experiments in isolated Python processes
# This prevents memory pool contamination between experiments

echo "========================================================================"
echo "Linear Layer Memory Profiling Experiment Suite"
echo "Each experiment runs in a fresh Python process to avoid contamination"
echo "========================================================================"
echo ""

# Configuration - Model Architecture
D=4096        # Input dimension
P=11008         # Output dimension
NUM_LAYERS=1    # Number of stacked Linear layers

# Configuration - Training
# Batch sizes to test (can be array for multiple values)
BATCH_SIZES=(8)
NUM_ITER=3
WARMUP_ITER=2
LEARNING_RATE=1e-5
SIGMA=1.0
MAX_GRAD_NORM=1.0

# Sequence lengths to test
SEQ_LENGTHS=(16384 32768)

# Modes to test
# Available Single-GPU modes: no_dp_single, grad_materialize, ghost, ghost_bk, flash, flash_bk, flash_fuse, flash_fuse_bk
# MODES=("no_dp_single" "grad_materialize" "ghost" "ghost_bk" "flash_fuse_bk")
# MODES=("grad_materialize" "ghost" "ghost_bk" "flash_fuse" "flash_fuse_bk" "no_dp_single")
MODES=("grad_materialize"  "flash_fuse" "no_dp_single") 
# Output directory
OUTPUT_DIR="results_linear"
mkdir -p "$OUTPUT_DIR"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$OUTPUT_DIR/run_$TIMESTAMP"
mkdir -p "$RUN_DIR"

echo "Output directory: $RUN_DIR"
echo "Model Architecture:"
echo "  - d (input dim): $D"
echo "  - p (output dim): $P"
echo "  - num_layers: $NUM_LAYERS"
echo ""
echo "Training Config:"
echo "  - Batch Sizes: ${BATCH_SIZES[@]}"
echo "  - Sequence Lengths: ${SEQ_LENGTHS[@]}"
echo "  - Modes: ${MODES[@]}"
echo ""

# Function to run a single experiment
run_experiment() {
    local mode=$1
    local seq_len=$2
    local batch_size=$3
    local output_file="$RUN_DIR/${mode}_seq${seq_len}_bs${batch_size}_result.json"
    
    echo "========================================================================"
    echo "Running: mode=$mode, seq_length=$seq_len, batch_size=$batch_size"
    echo "Output: $output_file"
    echo "========================================================================"
    
    # Activate virtual environment if needed
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi
    
    # Run experiment in isolated process
    python linear_experiment.py \
        --mode "$mode" \
        --seq-length $seq_len \
        --batch-size $batch_size \
        --num-iter $NUM_ITER \
        --warmup-iter $WARMUP_ITER \
        --d $D \
        --p $P \
        --num-layers $NUM_LAYERS \
        --learning-rate $LEARNING_RATE \
        --sigma $SIGMA \
        --max-grad-norm $MAX_GRAD_NORM \
        --output "$output_file"
    
    local exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        echo "❌ Experiment failed with exit code $exit_code"
        echo "Continuing with remaining experiments..."
    else
        echo "✓ Completed: mode=$mode, seq_length=$seq_len, batch_size=$batch_size"
    fi
    
    # Wait between experiments to ensure full cleanup
    echo "Waiting for cleanup..."
    sleep 5
    
    echo ""
}

# Run all experiments
echo "Starting experiment sequence..."
echo ""

total_experiments=$((${#MODES[@]} * ${#SEQ_LENGTHS[@]} * ${#BATCH_SIZES[@]}))
current_experiment=0

for batch_size in "${BATCH_SIZES[@]}"; do
    for seq_len in "${SEQ_LENGTHS[@]}"; do
        for mode in "${MODES[@]}"; do
            current_experiment=$((current_experiment + 1))
            echo ""
            echo "════════════════════════════════════════════════════════════════════"
            echo "Progress: Experiment $current_experiment / $total_experiments"
            echo "════════════════════════════════════════════════════════════════════"
            run_experiment "$mode" "$seq_len" "$batch_size"
        done
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
        --output-dir "$RUN_DIR/visualizations" \
        --baseline "no_dp_single"
    
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
printf "%-20s %-12s %-12s %15s %15s\n" "Mode" "Seq Length" "Batch Size" "Peak Mem (GB)" "Avg Time (ms)"
echo "------------------------------------------------------------------------------------"

# Print results for each experiment
for batch_size in "${BATCH_SIZES[@]}"; do
    for seq_len in "${SEQ_LENGTHS[@]}"; do
        for mode in "${MODES[@]}"; do
            result_file="$RUN_DIR/${mode}_seq${seq_len}_bs${batch_size}_result.json"
            if [ -f "$result_file" ]; then
                peak_mem=$(python -c "import json; data=json.load(open('$result_file')); print(f\"{data['peak_memory_gb']:.2f}\")" 2>/dev/null || echo "N/A")
                avg_time=$(python -c "import json; data=json.load(open('$result_file')); print(f\"{data['avg_time_ms']:.2f}\")" 2>/dev/null || echo "N/A")
                printf "%-20s %-12s %-12s %15s %15s\n" "$mode" "$seq_len" "$batch_size" "$peak_mem" "$avg_time"
            else
                printf "%-20s %-12s %-12s %15s %15s\n" "$mode" "$seq_len" "$batch_size" "FAILED" "FAILED"
            fi
        done
    done
    echo "------------------------------------------------------------------------------------"
done

echo ""
echo "✅ Complete pipeline finished!"
echo "✅ Results directory: $RUN_DIR"
if [ -d "$RUN_DIR/visualizations" ]; then
    echo "✅ Visualizations: $RUN_DIR/visualizations"
fi
echo ""

