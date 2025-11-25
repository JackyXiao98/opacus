#!/bin/bash
# Quick accuracy test - runs a few key modes to verify consistency
# This is a lighter version of run_accuracy_test.sh for quick checks

echo "========================================================================"
echo "Quick DPSGD Accuracy/Consistency Test"
echo "Testing key algorithms with epsilon≈1e-3, max_grad_norm=10000"
echo "========================================================================"
echo ""

# Check for HuggingFace token
if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: HF_TOKEN environment variable is not set"
    echo "Please set it with: export HF_TOKEN=your_token_here"
    return 1
fi

# Configuration - lighter for quick testing
MODEL_NAME="meta-llama/Llama-3.2-1B"
BATCH_SIZE=2
NUM_ITER=3           # Fewer iterations for quick test
WARMUP_ITER=1
VOCAB_SIZE=128256
LEARNING_RATE=1e-5
MAX_GRAD_NORM=10000
SEQ_LENGTH=1024
RANDOM_SEED=42
SIGMA=0.01

# Test only key modes
MODES=("no_dp" "flash_fsdp" "flash_fsdp_fuse")

# Output directory
OUTPUT_DIR="quick_accuracy_results"
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$OUTPUT_DIR/test_$TIMESTAMP"
mkdir -p "$RUN_DIR"

echo "Testing modes: ${MODES[@]}"
echo "Output: $RUN_DIR"
echo ""

# Activate virtual environment if needed
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Run tests
for mode in "${MODES[@]}"; do
    echo "Testing $mode..."
    python single_experiment.py \
        --mode "$mode" \
        --seq-length $SEQ_LENGTH \
        --batch-size $BATCH_SIZE \
        --num-iter $NUM_ITER \
        --warmup-iter $WARMUP_ITER \
        --output "$RUN_DIR/${mode}_result.json" \
        --model-name "$MODEL_NAME" \
        --token "$HF_TOKEN" \
        --vocab-size $VOCAB_SIZE \
        --learning-rate $LEARNING_RATE \
        --sigma $SIGMA \
        --max-grad-norm $MAX_GRAD_NORM \
        --accuracy-test \
        --random-seed $RANDOM_SEED
    
    if [ $? -eq 0 ]; then
        echo "✓ $mode completed"
    else
        echo "❌ $mode failed"
    fi
    
    sleep 3
    echo ""
done

# Quick comparison
echo "========================================================================"
echo "Quick Comparison:"
echo "========================================================================"
python -c "
import json
from pathlib import Path

results_dir = Path('$RUN_DIR')
modes = '$MODES'.split()

print('')
print(f\"{'Mode':<20} {'Final Loss':<15} {'Avg Grad Norm':<18}\")
print('-'*60)

for mode in modes:
    result_file = results_dir / f'{mode}_result.json'
    if result_file.exists():
        with open(result_file, 'r') as f:
            data = json.load(f)
            final_loss = data.get('final_loss', 0)
            avg_grad_norm = data.get('avg_grad_norm', 0)
            print(f\"{mode:<20} {final_loss:<15.6f} {avg_grad_norm:<18.6f}\")

print('-'*60)
print('')
"

echo "✅ Quick test complete!"
echo "Full results in: $RUN_DIR"

