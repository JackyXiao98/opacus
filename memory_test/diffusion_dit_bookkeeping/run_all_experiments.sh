#!/bin/bash
# Run all memory profiling experiments for DiT in isolated Python processes
# This prevents memory pool contamination between experiments

# set -e  # Exit on error

echo "========================================================================"
echo "DiT-L Memory Profiling Experiment Suite"
echo "Each experiment runs in a fresh Python process to avoid contamination"
echo "========================================================================"
echo ""

# Configuration - DiT-L with 1024 tokens
IMAGE_SIZE=256
PATCH_SIZE=8
IN_CHANNELS=3
NUM_CLASSES=1000
HIDDEN_DIM=1024
NUM_LAYERS=24
NUM_HEADS=16
BATCH_SIZE=2
NUM_ITER=3
WARMUP_ITER=0

# Compute number of tokens
NUM_TOKENS=$(( (IMAGE_SIZE / PATCH_SIZE) * (IMAGE_SIZE / PATCH_SIZE) ))

echo "Model Configuration:"
echo "  - Image Size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  - Patch Size: ${PATCH_SIZE}x${PATCH_SIZE}"
echo "  - Number of Tokens: ${NUM_TOKENS}"
echo "  - Hidden Dim: ${HIDDEN_DIM}"
echo "  - Number of Layers: ${NUM_LAYERS}"
echo "  - Number of Heads: ${NUM_HEADS}"
echo "  - Batch Size: ${BATCH_SIZE}"
echo ""

# Output directory
OUTPUT_DIR="memory_profiling_results"
mkdir -p "$OUTPUT_DIR"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$OUTPUT_DIR/run_$TIMESTAMP"
mkdir -p "$RUN_DIR"

echo "Output directory: $RUN_DIR"
echo ""

# Function to run a single experiment
run_experiment() {
    local exp_name=$1
    local output_file="$RUN_DIR/${exp_name}_result.json"
    
    echo "========================================================================"
    echo "Running: $exp_name"
    echo "Output: $output_file"
    echo "========================================================================"
    
    # Activate virtual environment if needed
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi
    
    # Run experiment in isolated process
    python single_experiment.py \
        --experiment "$exp_name" \
        --output "$output_file" \
        --image-size $IMAGE_SIZE \
        --patch-size $PATCH_SIZE \
        --in-channels $IN_CHANNELS \
        --num-classes $NUM_CLASSES \
        --hidden-dim $HIDDEN_DIM \
        --num-layers $NUM_LAYERS \
        --num-heads $NUM_HEADS \
        --batch-size $BATCH_SIZE \
        --num-iter $NUM_ITER \
        --warmup-iter $WARMUP_ITER
    
    # Wait a bit between experiments to ensure full cleanup
    sleep 3
    
    echo ""
    echo "✓ $exp_name completed"
    echo ""
}

# Run all experiments
echo "Starting experiment sequence..."
echo ""

# 1. Vanilla (baseline)
run_experiment "vanilla"

# 2. Flash Clipping (no bookkeeping)
run_experiment "flash_clip"

# 3. Flash Clipping w/ Bookkeeping
run_experiment "flash_clip_bookkeeping"

# 4. Ghost
run_experiment "ghost"

# 5. Bookkeeping
run_experiment "bookkeeping"

echo "========================================================================"
echo "All experiments completed!"
echo "========================================================================"
echo ""

# Generate visualization
echo "Generating visualizations..."
python visualize_memory_breakdown.py \
    --input-dir "$RUN_DIR" \
    --output-dir "$RUN_DIR/visualizations"

echo ""
echo "✅ Complete pipeline finished!"
echo "✅ Results directory: $RUN_DIR"
echo "✅ Visualizations: $RUN_DIR/visualizations"
echo ""

# Print summary
echo "========================================================================"
echo "SUMMARY"
echo "========================================================================"
for exp in vanilla ghost flash_clip flash_clip_bookkeeping bookkeeping; do
    result_file="$RUN_DIR/${exp}_result.json"
    if [ -f "$result_file" ]; then
        peak_mem=$(python -c "import json; data=json.load(open('$result_file')); print(f\"{data['peak_memory_mb']:.2f}\")")
        avg_time=$(python -c "import json; data=json.load(open('$result_file')); print(f\"{data['avg_time_ms']:.2f}\")")
        echo "  $exp: Peak Memory = $peak_mem MB, Avg Time = $avg_time ms"
    fi
done
echo "========================================================================"

