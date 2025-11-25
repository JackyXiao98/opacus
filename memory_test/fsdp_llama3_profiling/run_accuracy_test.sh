#!/bin/bash
# ==============================================================================
# Run accuracy/consistency test for DPSGD algorithms
# ==============================================================================
#
# PURPOSE:
#   This test verifies that different DPSGD algorithms (ghost_fsdp, flash_fsdp,
#   flash_fsdp_fuse, etc.) produce consistent results when privacy is minimal.
#
# TEST CONFIGURATION:
#   - epsilon ≈ 1e-3 (very low noise, minimal privacy)
#   - max_grad_norm = 10000 (very high, essentially no clipping)
#   - Fixed random seed = 42 (for reproducibility)
#
# WHAT IT MEASURES:
#   - Loss trajectory across iterations
#   - Gradient norms before optimizer step
#   - Linear layer weight norms (first and last iteration)
#
# EXPECTED OUTCOME:
#   All algorithms should produce very similar loss values and gradient norms,
#   demonstrating that they implement the same underlying DP-SGD algorithm
#   correctly, with differences only in efficiency/memory usage.
#
# USAGE:
#   export HF_TOKEN=your_huggingface_token
#   ./run_accuracy_test.sh
#
# OUTPUT:
#   - JSON files with detailed results for each mode
#   - Text summary with comparison table
#   - Plots showing loss trajectory and gradient norm comparison
#
# ==============================================================================

echo "========================================================================"
echo "DPSGD Accuracy/Consistency Test"
echo "Testing with epsilon≈1e-3, max_grad_norm=10000 (essentially no privacy)"
echo "All algorithms should produce similar loss and gradient norms"
echo "========================================================================"
echo ""

# Check for HuggingFace token
if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: HF_TOKEN environment variable is not set"
    echo "Please set it with: export HF_TOKEN=your_token_here"
    return 1
fi

# Configuration
MODEL_NAME="meta-llama/Llama-3.2-1B"
BATCH_SIZE=2
NUM_ITER=5           # More iterations to see convergence
WARMUP_ITER=1
VOCAB_SIZE=128256
LEARNING_RATE=1e-5
MAX_GRAD_NORM=10000   # Very high - essentially no clipping
SEQ_LENGTH=1024
RANDOM_SEED=42        # Fixed seed for reproducibility

# Calculate sigma from epsilon (approximately)
# For epsilon=1e-3 with delta=1e-5, batch_size=2*num_gpus, we need very low sigma
# Using RDP accountant approximation: sigma ≈ 0.01 gives epsilon ≈ 1e-3
SIGMA=0.01

# Modes to test - comparing different DPSGD algorithms
MODES=("no_dp" "ghost_fsdp" "flash_fsdp" "flash_fsdp_bk" "flash_fsdp_fuse" "flash_fsdp_fuse_bk")

# Output directory
OUTPUT_DIR="accuracy_test_results"
mkdir -p "$OUTPUT_DIR"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$OUTPUT_DIR/accuracy_test_$TIMESTAMP"
mkdir -p "$RUN_DIR"

echo "Output directory: $RUN_DIR"
echo "Model: $MODEL_NAME"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Sequence length: $SEQ_LENGTH"
echo "Iterations: $NUM_ITER"
echo "Sigma: $SIGMA (epsilon ≈ 1e-3)"
echo "Max grad norm: $MAX_GRAD_NORM"
echo "Random seed: $RANDOM_SEED"
echo "Modes to test: ${MODES[@]}"
echo ""

# Function to run a single accuracy test
run_accuracy_test() {
    local mode=$1
    local output_file="$RUN_DIR/${mode}_accuracy_result.json"
    
    echo "========================================================================"
    echo "Testing: mode=$mode"
    echo "Output: $output_file"
    echo "========================================================================"
    
    # Activate virtual environment if needed
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi
    
    # Run accuracy test
    python single_experiment.py \
        --mode "$mode" \
        --seq-length $SEQ_LENGTH \
        --batch-size $BATCH_SIZE \
        --num-iter $NUM_ITER \
        --warmup-iter $WARMUP_ITER \
        --output "$output_file" \
        --model-name "$MODEL_NAME" \
        --token "$HF_TOKEN" \
        --vocab-size $VOCAB_SIZE \
        --learning-rate $LEARNING_RATE \
        --sigma $SIGMA \
        --max-grad-norm $MAX_GRAD_NORM \
        --accuracy-test \
        --random-seed $RANDOM_SEED
    
    local exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        echo "❌ Test failed with exit code $exit_code"
        echo "Continuing with remaining tests..."
    else
        echo "✓ Completed: mode=$mode"
    fi
    
    # Wait between tests
    echo "Waiting for cleanup..."
    sleep 5
    
    echo ""
}

# Run all tests
echo "Starting accuracy tests..."
echo ""

total_tests=${#MODES[@]}
current_test=0

for mode in "${MODES[@]}"; do
    current_test=$((current_test + 1))
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "Progress: Test $current_test / $total_tests"
    echo "════════════════════════════════════════════════════════════════════"
    run_accuracy_test "$mode"
done

echo "========================================================================"
echo "All accuracy tests completed!"
echo "========================================================================"
echo ""

# Generate comparison summary
echo "Generating accuracy comparison summary..."
python -c "
import json
import sys
from pathlib import Path

results_dir = Path('$RUN_DIR')
modes = '$MODES'.split()

print('')
print('='*100)
print('ACCURACY TEST RESULTS - Algorithm Consistency Comparison')
print('='*100)
print('')
print(f'Configuration: sigma={$SIGMA}, max_grad_norm={$MAX_GRAD_NORM}, seed={$RANDOM_SEED}')
print('')

# Load all results
all_results = {}
for mode in modes:
    result_file = results_dir / f'{mode}_accuracy_result.json'
    if result_file.exists():
        with open(result_file, 'r') as f:
            all_results[mode] = json.load(f)

if not all_results:
    print('❌ No results found!')
    sys.exit(1)

# Print loss values comparison
print('-'*100)
print(f\"{'Mode':<25} {'Final Loss':<15} {'Loss Trajectory':<60}\")
print('-'*100)

for mode in modes:
    if mode in all_results:
        data = all_results[mode]
        final_loss = data.get('final_loss', 0)
        loss_values = data.get('loss_values', [])
        loss_str = ', '.join([f'{v:.6f}' for v in loss_values])
        print(f\"{mode:<25} {final_loss:<15.6f} {loss_str:<60}\")

print('-'*100)
print('')

# Print gradient norms comparison
print('-'*100)
print(f\"{'Mode':<25} {'Avg Grad Norm':<20} {'Grad Norm Trajectory':<55}\")
print('-'*100)

for mode in modes:
    if mode in all_results:
        data = all_results[mode]
        avg_grad_norm = data.get('avg_grad_norm', 0)
        grad_norms = data.get('grad_norms', [])
        if grad_norms:
            grad_str = ', '.join([f'{v:.6f}' for v in grad_norms])
            print(f\"{mode:<25} {avg_grad_norm:<20.6f} {grad_str:<55}\")
        else:
            print(f\"{mode:<25} {'N/A':<20} {'N/A':<55}\")

print('-'*100)
print('')

# Calculate consistency metrics
print('CONSISTENCY ANALYSIS:')
print('')

# Compare final losses
final_losses = [all_results[mode]['final_loss'] for mode in modes if mode in all_results]
if len(final_losses) > 1:
    avg_loss = sum(final_losses) / len(final_losses)
    max_loss_diff = max(final_losses) - min(final_losses)
    loss_std = (sum([(l - avg_loss)**2 for l in final_losses]) / len(final_losses)) ** 0.5
    print(f'  Final Loss - Mean: {avg_loss:.6f}, Std: {loss_std:.6f}, Max Diff: {max_loss_diff:.6f}')

# Compare gradient norms
grad_norm_avgs = [all_results[mode].get('avg_grad_norm', 0) for mode in modes if mode in all_results and all_results[mode].get('avg_grad_norm', 0) > 0]
if len(grad_norm_avgs) > 1:
    avg_grad_norm = sum(grad_norm_avgs) / len(grad_norm_avgs)
    max_grad_diff = max(grad_norm_avgs) - min(grad_norm_avgs)
    grad_std = (sum([(g - avg_grad_norm)**2 for g in grad_norm_avgs]) / len(grad_norm_avgs)) ** 0.5
    print(f'  Avg Grad Norm - Mean: {avg_grad_norm:.6f}, Std: {grad_std:.6f}, Max Diff: {max_grad_diff:.6f}')

print('')
print('='*100)

# Save summary to file
summary_file = results_dir / 'accuracy_summary.txt'
print(f'Summary saved to: {summary_file}')
" | tee "$RUN_DIR/accuracy_summary.txt"

echo ""
echo "✅ Accuracy test pipeline completed!"
echo "✅ Results directory: $RUN_DIR"
echo "✅ Summary: $RUN_DIR/accuracy_summary.txt"
echo ""

