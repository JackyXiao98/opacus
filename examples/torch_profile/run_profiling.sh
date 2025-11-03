#!/bin/bash

echo "Starting GPU profiling run with multiple independent Python processes..."

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3."
fi

# Configuration arrays
TRAINERS=("StandardTrainer" "DPSGDTrainer" "DPGhostClippingTrainer" "DPFastGradientClippingTrainer")
BATCH_SIZES=(4 8)
SEQ_LENGTHS=(512 1024)
MODEL_SIZE="1b"

# Create logs directory
mkdir -p logs

# Function to run a single configuration
run_single_config() {
    local trainer=$1
    local batch_size=$2
    local seq_len=$3
    local log_file="logs/${trainer}_bs${batch_size}_seq${seq_len}.log"
    
    echo "🚀 Starting: $trainer (batch_size=$batch_size, seq_len=$seq_len)"
    echo "   Log file: $log_file"
    
    # Run in a fresh Python process
    if python3 profiling_script.py --mode=single \
        --trainer="$trainer" \
        --batch-size="$batch_size" \
        --seq-len="$seq_len" \
        --model-size="$MODEL_SIZE" > "$log_file" 2>&1; then
        echo "✅ Completed: $trainer (batch_size=$batch_size, seq_len=$seq_len)"
        return 0
    else
        echo "❌ Failed: $trainer (batch_size=$batch_size, seq_len=$seq_len)"
        echo "   Check log file: $log_file"
        return 1
    fi
}

# Track results
total_configs=0
successful_configs=0
failed_configs=0

echo "Configuration matrix:"
echo "  Trainers: ${TRAINERS[*]}"
echo "  Batch sizes: ${BATCH_SIZES[*]}"
echo "  Sequence lengths: ${SEQ_LENGTHS[*]}"
echo "  Model size: $MODEL_SIZE"
echo ""

# Run all configurations
for trainer in "${TRAINERS[@]}"; do
    echo "=== Testing $trainer ==="
    
    for batch_size in "${BATCH_SIZES[@]}"; do
        for seq_len in "${SEQ_LENGTHS[@]}"; do
            total_configs=$((total_configs + 1))
            
            if run_single_config "$trainer" "$batch_size" "$seq_len"; then
                successful_configs=$((successful_configs + 1))
            else
                failed_configs=$((failed_configs + 1))
            fi
            
            # Small delay between runs to ensure clean separation
            sleep 2
        done
    done
    
    echo "Completed all configurations for $trainer"
    echo ""
done

# Summary
echo "========================================="
echo "🎉 Profiling run completed!"
echo "========================================="
echo "Total configurations: $total_configs"
echo "Successful: $successful_configs"
echo "Failed: $failed_configs"
echo ""

if [ $failed_configs -gt 0 ]; then
    echo "⚠️  Some configurations failed. Check log files in ./logs/ directory"
    echo "Failed configurations:"
    grep -l "ERROR\|Failed\|Exception" logs/*.log 2>/dev/null | sed 's/^/  - /'
fi

echo ""
echo "📊 View results with: tensorboard --logdir=./runs"
echo "📋 Log files are in: ./logs/"
