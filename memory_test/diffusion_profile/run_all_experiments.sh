#!/bin/bash
# Run all memory profiling experiments for DiT (facebook/DiT-XL-2-256) in isolated Python processes
# This prevents memory pool contamination between experiments

# set -e  # Exit on error

# Default configuration
IMAGE_SIZE=256
PATCH_SIZE=2
IN_CHANNELS=4
NUM_CLASSES=1000
BATCH_SIZE=1
NUM_ITER=1
WARMUP_ITER=1
USE_FLASH_ATTENTION="--use-flash-attention"  # Set to "" to disable, or "--use-flash-attention" to enable
EXPERIMENTS="vanilla ghost flash_clip flash_clip_bk bookkeeping"  # Default: run all

# Parse command line arguments
USAGE="Usage: $0 [OPTIONS]

Model Configuration:
  --image-size SIZE          Image size (default: 256)
  --patch-size SIZE          Patch size (default: 2)
  --in-channels N            Number of input channels (default: 4)
  --num-classes N            Number of classes (default: 1000)

Training Configuration:
  --batch-size N             Batch size (default: 1)
  --num-iter N               Number of iterations (default: 1)
  --warmup-iter N            Number of warmup iterations (default: 1)

Optimization Options:
  --use-flash-attention      Enable Flash Attention for memory efficiency (default: enabled)
  --no-flash-attention       Disable Flash Attention

Experiment Selection:
  --experiments EXP1,EXP2    Run specific experiments (comma-separated)
                            Available: vanilla, ghost, flash_clip, flash_clip_bk, bookkeeping, ghost_fsdp_bk
                            Default: all experiments
  --help                     Display this help message

Examples:
  # Run all experiments with default settings
  $0

  # Run only vanilla and ghost experiments
  $0 --experiments vanilla,ghost

  # Run without Flash Attention
  $0 --no-flash-attention

  # Run with larger batch size
  $0 --batch-size 4 --num-iter 10
"

while [[ $# -gt 0 ]]; do
    case $1 in
        --image-size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        --patch-size)
            PATCH_SIZE="$2"
            shift 2
            ;;
        --in-channels)
            IN_CHANNELS="$2"
            shift 2
            ;;
        --num-classes)
            NUM_CLASSES="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-iter)
            NUM_ITER="$2"
            shift 2
            ;;
        --warmup-iter)
            WARMUP_ITER="$2"
            shift 2
            ;;
        --use-flash-attention)
            USE_FLASH_ATTENTION="--use-flash-attention"
            shift
            ;;
        --no-flash-attention)
            USE_FLASH_ATTENTION=""
            shift
            ;;
        --experiments)
            # Convert comma-separated list to space-separated
            EXPERIMENTS=$(echo "$2" | tr ',' ' ')
            shift 2
            ;;
        --help)
            echo "$USAGE"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "$USAGE"
            exit 1
            ;;
    esac
done

echo "========================================================================"
echo "DiT (facebook/DiT-XL-2-256) Memory Profiling Experiment Suite"
echo "Each experiment runs in a fresh Python process to avoid contamination"
echo "========================================================================"
echo ""

# Compute number of tokens
NUM_TOKENS=$(( (IMAGE_SIZE / PATCH_SIZE) * (IMAGE_SIZE / PATCH_SIZE) ))

echo "Model Configuration:"
echo "  - Model: facebook/DiT-XL-2-256 (via diffusers)"
echo "  - Image Size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  - Patch Size: ${PATCH_SIZE}x${PATCH_SIZE}"
echo "  - Input Channels: ${IN_CHANNELS}"
echo "  - Number of Tokens: ${NUM_TOKENS}"
echo "  - Batch Size: ${BATCH_SIZE}"
echo "  - Num Iterations: ${NUM_ITER}"
echo "  - Warmup Iterations: ${WARMUP_ITER}"
if [ -n "$USE_FLASH_ATTENTION" ]; then
    echo "  - Flash Attention: Enabled"
else
    echo "  - Flash Attention: Disabled"
fi
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
        --batch-size $BATCH_SIZE \
        --num-iter $NUM_ITER \
        --warmup-iter $WARMUP_ITER \
        $USE_FLASH_ATTENTION
    
    local exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        echo "⚠️  $exp_name failed with exit code $exit_code"
        echo "   Continuing with next experiment..."
    fi
    
    # Wait a bit between experiments to ensure full cleanup
    sleep 3
    
    echo ""
    echo "✓ $exp_name completed"
    echo ""
}

# Run selected experiments
echo "Starting experiment sequence..."
echo "Experiments to run: $EXPERIMENTS"
echo ""

# Run each experiment
for exp in $EXPERIMENTS; do
    case $exp in
        vanilla|ghost|flash_clip|flash_clip_bk|bookkeeping|ghost_fsdp_bk)
            run_experiment "$exp"
            ;;
        *)
            echo "⚠️  Unknown experiment: $exp (skipping)"
            echo ""
            ;;
    esac
done

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
echo ""
printf "%-30s %-20s %-20s\n" "Experiment" "Peak Memory (MB)" "Avg Time (ms)"
printf "%-30s %-20s %-20s\n" "----------" "---------------" "-------------"
for exp in $EXPERIMENTS; do
    result_file="$RUN_DIR/${exp}_result.json"
    if [ -f "$result_file" ]; then
        peak_mem=$(python -c "import json; data=json.load(open('$result_file')); print(f\"{data['peak_memory_mb']:.2f}\")" 2>/dev/null || echo "N/A")
        avg_time=$(python -c "import json; data=json.load(open('$result_file')); print(f\"{data['avg_time_ms']:.2f}\")" 2>/dev/null || echo "N/A")
        printf "%-30s %-20s %-20s\n" "$exp" "$peak_mem" "$avg_time"
    else
        printf "%-30s %-20s %-20s\n" "$exp" "FAILED/SKIPPED" "-"
    fi
done
echo ""
echo "========================================================================"


