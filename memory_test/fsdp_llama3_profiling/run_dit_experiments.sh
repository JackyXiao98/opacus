#!/bin/bash
# Run all FSDP DiT memory profiling experiments in isolated Python processes
# This prevents memory pool contamination between experiments

echo "========================================================================"
echo "FSDP DiT Memory Profiling Experiment Suite"
echo "Each experiment runs in a fresh Python process to avoid contamination"
echo "========================================================================"
echo ""

# DiT Model Configuration
IMAGE_SIZE=256
PATCH_SIZE=8
IN_CHANNELS=3
NUM_CLASSES=1000
HIDDEN_DIM=1152
NUM_LAYERS=28
NUM_HEADS=16

# Training Configuration
BATCH_SIZE=1
NUM_ITER=3
WARMUP_ITER=1
LEARNING_RATE=1e-4
SIGMA=1.0
MAX_GRAD_NORM=1.0

# Compute number of tokens (sequence length for DiT)
NUM_TOKENS=$(( (IMAGE_SIZE / PATCH_SIZE) * (IMAGE_SIZE / PATCH_SIZE) ))

# Modes to test
# Available FSDP modes: no_dp, ghost_fsdp, flash_fsdp, flash_fsdp_bk, ghost_fsdp_bk, flash_fsdp_fuse, flash_fsdp_fuse_bk
# Available Single-GPU modes: no_dp_single, ghost, flash, flash_bk, ghost_bk, flash_fuse, flash_fuse_bk
MODES=("no_dp" "ghost_fsdp" "flash_fsdp" "flash_fsdp_bk" "flash_fsdp_fuse")

# Output directory
OUTPUT_DIR="results_dit"
mkdir -p "$OUTPUT_DIR"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$OUTPUT_DIR/run_$TIMESTAMP"
mkdir -p "$RUN_DIR"

echo "Model Configuration:"
echo "  - Image Size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  - Patch Size: ${PATCH_SIZE}x${PATCH_SIZE}"
echo "  - Number of Tokens: ${NUM_TOKENS}"
echo "  - Hidden Dim: ${HIDDEN_DIM}"
echo "  - Number of Layers: ${NUM_LAYERS}"
echo "  - Number of Heads: ${NUM_HEADS}"
echo "  - Batch Size: ${BATCH_SIZE}"
echo ""
echo "Output directory: $RUN_DIR"
echo "Modes: ${MODES[@]}"
echo ""

# Function to run a single experiment
run_experiment() {
    local mode=$1
    local output_file="$RUN_DIR/${mode}_seq${NUM_TOKENS}_result.json"
    
    echo "========================================================================"
    echo "Running: mode=$mode, num_tokens=$NUM_TOKENS"
    echo "Output: $output_file"
    echo "========================================================================"
    
    # Activate virtual environment if needed
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi
    
    # Run experiment in isolated process
    python single_experiment.py \
        --model-type dit \
        --mode "$mode" \
        --batch-size $BATCH_SIZE \
        --num-iter $NUM_ITER \
        --warmup-iter $WARMUP_ITER \
        --output "$output_file" \
        --image-size $IMAGE_SIZE \
        --patch-size $PATCH_SIZE \
        --in-channels $IN_CHANNELS \
        --hidden-dim $HIDDEN_DIM \
        --num-layers $NUM_LAYERS \
        --num-heads $NUM_HEADS \
        --num-classes $NUM_CLASSES \
        --learning-rate $LEARNING_RATE \
        --sigma $SIGMA \
        --max-grad-norm $MAX_GRAD_NORM
    
    local exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        echo "❌ Experiment failed with exit code $exit_code"
        echo "Continuing with remaining experiments..."
    else
        echo "✓ Completed: mode=$mode, num_tokens=$NUM_TOKENS"
    fi
    
    # Wait between experiments to ensure full cleanup
    echo "Waiting for cleanup..."
    sleep 5
    
    echo ""
}

# Run all experiments
echo "Starting experiment sequence..."
echo ""

total_experiments=${#MODES[@]}
current_experiment=0

for mode in "${MODES[@]}"; do
    current_experiment=$((current_experiment + 1))
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "Progress: Experiment $current_experiment / $total_experiments"
    echo "════════════════════════════════════════════════════════════════════"
    run_experiment "$mode"
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
printf "%-20s %-12s %15s %15s\n" "Mode" "Num Tokens" "Peak Mem (GB)" "Avg Time (ms)"
echo "------------------------------------------------------------------------"

# Print results for each experiment
for mode in "${MODES[@]}"; do
    result_file="$RUN_DIR/${mode}_seq${NUM_TOKENS}_result.json"
    if [ -f "$result_file" ]; then
        peak_mem=$(python -c "import json; data=json.load(open('$result_file')); print(f\"{data['peak_memory_gb']:.2f}\")" 2>/dev/null || echo "N/A")
        avg_time=$(python -c "import json; data=json.load(open('$result_file')); print(f\"{data['avg_time_ms']:.2f}\")" 2>/dev/null || echo "N/A")
        printf "%-20s %-12s %15s %15s\n" "$mode" "$NUM_TOKENS" "$peak_mem" "$avg_time"
    else
        printf "%-20s %-12s %15s %15s\n" "$mode" "$NUM_TOKENS" "FAILED" "FAILED"
    fi
done
echo "------------------------------------------------------------------------"

echo ""
echo "✅ Complete pipeline finished!"
echo "✅ Results directory: $RUN_DIR"
if [ -d "$RUN_DIR/visualizations" ]; then
    echo "✅ Visualizations: $RUN_DIR/visualizations"
fi
echo ""

