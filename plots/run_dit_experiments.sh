#!/bin/bash
# Run all FSDP DiT memory profiling experiments in isolated Python processes
# This prevents memory pool contamination between experiments

echo "========================================================================"
echo "FSDP DiT Memory Profiling Experiment Suite"
echo "Each experiment runs in a fresh Python process to avoid contamination"
echo "========================================================================"
echo ""

# DiT Model Configuration (uses same model as dp-train.py)
# Available models: DiT-XL/2, DiT-XL/4, DiT-XL/8, DiT-L/2, DiT-L/4, DiT-L/8, DiT-B/2, DiT-B/4, DiT-B/8, DiT-S/2, DiT-S/4, DiT-S/8
DIT_MODEL_NAME="DiT-S/2"
IMAGE_SIZES=(128 256 512)
IN_CHANNELS=2
NUM_CLASSES=1000

# Training Configuration
BATCH_SIZE=2
NUM_ITER=3
WARMUP_ITER=2
LEARNING_RATE=1e-4
SIGMA=1.0
MAX_GRAD_NORM=1.0

# Modes to test
# Available FSDP modes: no_dp, ghost_fsdp, flash_fsdp, flash_fsdp_bk, ghost_fsdp_bk, flash_fsdp_fuse, flash_fsdp_fuse_bk
# Available Single-GPU modes: no_dp_single, ghost, flash, flash_bk, ghost_bk, flash_fuse, flash_fuse_bk
# MODES=("flash_fuse_bk" "no_dp_single" "flash_bk")
# "no_dp_single" "ghost" "flash" "flash_bk" "ghost_bk" "flash_fuse" 
MODES=("flash_fsdp_fuse_bk" "no_dp" "flash_fsdp_fuse" "flash_fsdp_bk")
# MODES=("flash_fuse_bk" "no_dp_single" "flash_fuse" "flash_bk")
# Output directory
OUTPUT_DIR="results_dit"
mkdir -p "$OUTPUT_DIR"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$OUTPUT_DIR/run_$TIMESTAMP"
mkdir -p "$RUN_DIR"

echo "Model Configuration:"
echo "  - DiT Model: ${DIT_MODEL_NAME}"
echo "  - Image Sizes: ${IMAGE_SIZES[*]}"
echo "  - Patch Size: $(echo "$DIT_MODEL_NAME" | grep -oE '[0-9]+$')"
echo "  - Batch Size: ${BATCH_SIZE}"
echo ""
echo "Output directory: $RUN_DIR"
echo "Modes: ${MODES[@]}"
echo ""

# Function to run a single experiment
run_experiment() {
    local mode=$1
    local image_size=$2
    local patch_size
    patch_size=$(echo "$DIT_MODEL_NAME" | grep -oE '[0-9]+$')
    local latent_size=$((image_size / 8))
    local seq_length=$(((latent_size / patch_size) * (latent_size / patch_size)))
    local output_file="$RUN_DIR/${mode}_img${image_size}_seq${seq_length}_result.json"
    
    echo "========================================================================"
    echo "Running: mode=$mode, image_size=${image_size}, seq_length=${seq_length}"
    echo "Output: $output_file"
    echo "========================================================================"
    
    # Activate virtual environment if needed
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi
    
    # Run experiment in isolated process (uses same model as dp-train.py)
    python dit_experiment.py \
        --mode "$mode" \
        --batch-size $BATCH_SIZE \
        --num-iter $NUM_ITER \
        --warmup-iter $WARMUP_ITER \
        --output "$output_file" \
        --dit-model-name "$DIT_MODEL_NAME" \
        --image-size $image_size \
        --seq-length $seq_length \
        --in-channels $IN_CHANNELS \
        --num-classes $NUM_CLASSES \
        --learning-rate $LEARNING_RATE \
        --sigma $SIGMA \
        --max-grad-norm $MAX_GRAD_NORM
    
    local exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        echo "❌ Experiment failed with exit code $exit_code"
        echo "Continuing with remaining experiments..."
    else
        echo "✓ Completed: mode=$mode, seq_length=${seq_length}"
    fi
    
    # Wait between experiments to ensure full cleanup
    echo "Waiting for cleanup..."
    sleep 5
    
    echo ""
}

# Run all experiments
echo "Starting experiment sequence..."
echo ""

total_experiments=$(( ${#MODES[@]} * ${#IMAGE_SIZES[@]} ))
current_experiment=0

for image_size in "${IMAGE_SIZES[@]}"; do
    patch_size=$(echo "$DIT_MODEL_NAME" | grep -oE '[0-9]+$')
    latent_size=$((image_size / 8))
    seq_length=$(((latent_size / patch_size) * (latent_size / patch_size)))

    echo ""
    echo "--------------------------------------------------------------------"
    echo "Image size: ${image_size} | Latent: ${latent_size} | Seq Length: ${seq_length}"
    echo "--------------------------------------------------------------------"

    for mode in "${MODES[@]}"; do
        current_experiment=$((current_experiment + 1))
        echo ""
        echo "════════════════════════════════════════════════════════════════════"
        echo "Progress: Experiment $current_experiment / $total_experiments"
        echo "════════════════════════════════════════════════════════════════════"
        run_experiment "$mode" "$image_size"
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
printf "%-20s %-12s %15s %15s\n" "Mode" "Num Tokens" "Peak Mem (GB)" "Avg Time (ms)"
echo "------------------------------------------------------------------------"

# Print results for each experiment
for image_size in "${IMAGE_SIZES[@]}"; do
    patch_size=$(echo "$DIT_MODEL_NAME" | grep -oE '[0-9]+$')
    latent_size=$((image_size / 8))
    seq_length=$(((latent_size / patch_size) * (latent_size / patch_size)))

    for mode in "${MODES[@]}"; do
        result_file="$RUN_DIR/${mode}_img${image_size}_seq${seq_length}_result.json"
        if [ -f "$result_file" ]; then
            peak_mem=$(python -c "import json; data=json.load(open('$result_file')); print(f\"{data['peak_memory_gb']:.2f}\")" 2>/dev/null || echo "N/A")
            avg_time=$(python -c "import json; data=json.load(open('$result_file')); print(f\"{data['avg_time_ms']:.2f}\")" 2>/dev/null || echo "N/A")
            printf "%-20s %-12s %15s %15s\n" "$mode" "$seq_length" "$peak_mem" "$avg_time"
        else
            printf "%-20s %-12s %15s %15s\n" "$mode" "$seq_length" "FAILED" "FAILED"
        fi
    done
done
echo "------------------------------------------------------------------------"

echo ""
echo "✅ Complete pipeline finished!"
echo "✅ Results directory: $RUN_DIR"
if [ -d "$RUN_DIR/visualizations" ]; then
    echo "✅ Visualizations: $RUN_DIR/visualizations"
fi
echo ""

