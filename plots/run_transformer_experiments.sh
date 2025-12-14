#!/bin/bash
# Run all custom Transformer model profiling experiments in isolated Python processes
# This prevents memory pool contamination between experiments

echo "========================================================================"
echo "Custom Transformer Memory Profiling Experiment Suite"
echo "Each experiment runs in a fresh Python process to avoid contamination"
echo "========================================================================"
echo ""

# Configuration - Model Architecture
VOCAB_SIZE=50257        # GPT-2 的词表大小
HIDDEN_DIM=1280         # GPT-2 Large 的 hidden size
NUM_LAYERS=36           # 36 层 transformer block
NUM_HEADS=20            # 20 个 attention heads

# Configuration - Training
BATCH_SIZE=4
NUM_ITER=3
WARMUP_ITER=2
LEARNING_RATE=1e-5
SIGMA=1.0
MAX_GRAD_NORM=1.0

# Sequence lengths to test
SEQ_LENGTHS=(512 1024 2048 4096 8192)

# Modes to test
# Available Single-GPU modes: no_dp_single, grad_materialize, ghost, ghost_bk, flash, flash_bk, flash_fuse, flash_fuse_bk
# Available FSDP modes: no_dp, ghost_fsdp, flash_fsdp, flash_fsdp_bk, ghost_fsdp_bk, flash_fsdp_fuse, flash_fsdp_fuse_bk
# MODES=("flash_fuse_bk" "flash_bk" "no_dp_single")
# MODES=("no_dp_single" "grad_materialize" "ghost_bk" "flash_fuse_bk")
MODES=("grad_materialize" "ghost" "ghost_bk" "flash_fuse" "flash_fuse_bk" "no_dp_single")

# Output directory
OUTPUT_DIR="results_transformer"
mkdir -p "$OUTPUT_DIR"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$OUTPUT_DIR/run_$TIMESTAMP"
mkdir -p "$RUN_DIR"

echo "Output directory: $RUN_DIR"
echo "Model Architecture:"
echo "  - Vocab Size: $VOCAB_SIZE"
echo "  - Hidden Dim: $HIDDEN_DIM"
echo "  - Num Layers: $NUM_LAYERS"
echo "  - Num Heads: $NUM_HEADS"
echo ""
echo "Training Config:"
echo "  - Batch Size per GPU: $BATCH_SIZE"
echo "  - Sequence Lengths: ${SEQ_LENGTHS[@]}"
echo "  - Modes: ${MODES[@]}"
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
    
    # Run experiment in isolated process
    python transformer_experiment.py \
        --mode "$mode" \
        --seq-length $seq_len \
        --batch-size $BATCH_SIZE \
        --num-iter $NUM_ITER \
        --warmup-iter $WARMUP_ITER \
        --bf16 \
        --vocab-size $VOCAB_SIZE \
        --hidden-dim $HIDDEN_DIM \
        --num-layers $NUM_LAYERS \
        --num-heads $NUM_HEADS \
        --learning-rate $LEARNING_RATE \
        --sigma $SIGMA \
        --max-grad-norm $MAX_GRAD_NORM \
        --output "$output_file"
    
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
printf "%-20s %-12s %15s %15s\n" "Mode" "Seq Length" "Peak Mem (GB)" "Avg Time (ms)"
echo "------------------------------------------------------------------------"

# Print results for each experiment
for seq_len in "${SEQ_LENGTHS[@]}"; do
    for mode in "${MODES[@]}"; do
        result_file="$RUN_DIR/${mode}_seq${seq_len}_result.json"
        if [ -f "$result_file" ]; then
            peak_mem=$(python -c "import json; data=json.load(open('$result_file')); print(f\"{data['peak_memory_gb']:.2f}\")" 2>/dev/null || echo "N/A")
            avg_time=$(python -c "import json; data=json.load(open('$result_file')); print(f\"{data['avg_time_ms']:.2f}\")" 2>/dev/null || echo "N/A")
            printf "%-20s %-12s %15s %15s\n" "$mode" "$seq_len" "$peak_mem" "$avg_time"
        else
            printf "%-20s %-12s %15s %15s\n" "$mode" "$seq_len" "FAILED" "FAILED"
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

