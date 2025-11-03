#!/bin/bash

echo "Testing multi-process profiling with small configurations..."

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3."
    exit 1
fi

# Small test configuration
TRAINERS=("StandardTrainer" "DPSGDTrainer")
BATCH_SIZES=(2)
SEQ_LENGTHS=(256)
MODEL_SIZE="tiny"

# Create logs directory
mkdir -p logs

# Function to run a single configuration
run_single_config() {
    local trainer=$1
    local batch_size=$2
    local seq_len=$3
    local log_file="logs/test_${trainer}_bs${batch_size}_seq${seq_len}.log"
    
    echo "🚀 Testing: $trainer (batch_size=$batch_size, seq_len=$seq_len)"
    
    # Run in a fresh Python process
    if python3 profiling_script.py --mode=single \
        --trainer="$trainer" \
        --batch-size="$batch_size" \
        --seq-len="$seq_len" \
        --model-size="$MODEL_SIZE" > "$log_file" 2>&1; then
        echo "✅ Success: $trainer"
        return 0
    else
        echo "❌ Failed: $trainer (check $log_file)"
        return 1
    fi
}

# Track results
total_configs=0
successful_configs=0

echo "Test configuration:"
echo "  Trainers: ${TRAINERS[*]}"
echo "  Batch size: ${BATCH_SIZES[*]}"
echo "  Sequence length: ${SEQ_LENGTHS[*]}"
echo "  Model size: $MODEL_SIZE"
echo ""

# Run test configurations
for trainer in "${TRAINERS[@]}"; do
    for batch_size in "${BATCH_SIZES[@]}"; do
        for seq_len in "${SEQ_LENGTHS[@]}"; do
            total_configs=$((total_configs + 1))
            
            if run_single_config "$trainer" "$batch_size" "$seq_len"; then
                successful_configs=$((successful_configs + 1))
            fi
            
            # Small delay between runs
            sleep 1
        done
    done
done

echo ""
echo "========================================="
echo "🎉 Test completed!"
echo "========================================="
echo "Total: $total_configs, Successful: $successful_configs"

if [ $successful_configs -eq $total_configs ]; then
    echo "✅ All tests passed! The multi-process approach is working correctly."
    exit 0
else
    echo "❌ Some tests failed. Check log files in ./logs/"
    exit 1
fi