#!/bin/bash

# Single GPU Memory Analysis Script for Opacus
# This script runs memory profiling on a single GPU with different configurations

set -e

# Default values
TOKEN=""
MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
RESULTS_DIR="./single_gpu_memory_results"
NUM_STEPS=5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --token)
            TOKEN="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --results_dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --num_steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --token TOKEN           Hugging Face token (required for gated models)"
            echo "  --model_name MODEL      Model name (default: meta-llama/Llama-3.2-3B-Instruct)"
            echo "  --results_dir DIR       Results directory (default: ./single_gpu_memory_results)"
            echo "  --num_steps STEPS       Number of training steps (default: 5)"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if GPU is available
if ! python -c "import torch; print('CUDA available:', torch.cuda.is_available())" | grep -q "True"; then
    echo "Error: CUDA is not available. This script requires a GPU."
    exit 1
fi

echo "=== Single GPU Memory Analysis ==="
echo "Model: $MODEL_NAME"
echo "Results directory: $RESULTS_DIR"
echo "Training steps: $NUM_STEPS"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Function to run profiling with specific configuration
run_profile() {
    local seq_length=$1
    local batch_size=$2
    local max_physical_batch_size=$3
    local use_lora=$4
    local lora_rank=$5
    local config_name=$6
    
    echo "Running configuration: $config_name"
    echo "  Sequence length: $seq_length"
    echo "  Batch size: $batch_size"
    echo "  Max physical batch size: $max_physical_batch_size"
    echo "  Use LoRA: $use_lora"
    echo "  LoRA rank: $lora_rank"
    
    local cmd="python single_gpu_memory_profiler.py \
        --model_name \"$MODEL_NAME\" \
        --seq_length $seq_length \
        --batch_size $batch_size \
        --max_physical_batch_size $max_physical_batch_size \
        --num_steps $NUM_STEPS \
        --results_dir \"$RESULTS_DIR/$config_name\""
    
    if [ "$TOKEN" != "" ]; then
        cmd="$cmd --token \"$TOKEN\""
    fi
    
    if [ "$use_lora" = "true" ]; then
        cmd="$cmd --use_lora --lora_rank $lora_rank"
    fi
    
    echo "Command: $cmd"
    echo ""
    
    eval $cmd
    
    if [ $? -eq 0 ]; then
        echo "✓ Configuration $config_name completed successfully"
    else
        echo "✗ Configuration $config_name failed"
        return 1
    fi
    echo ""
}

# Basic configuration - small model test
echo "=== Running Basic Configuration ==="
run_profile 128 8 1 false 16 "basic_config"

# Test with different sequence lengths
echo "=== Testing Different Sequence Lengths ==="
run_profile 64 8 1 false 16 "seq_64"
run_profile 256 8 1 false 16 "seq_256"

# Test with different batch sizes
echo "=== Testing Different Batch Sizes ==="
run_profile 128 16 1 false 16 "batch_16"
run_profile 128 32 1 false 16 "batch_32"

# Test with LoRA
echo "=== Testing with LoRA ==="
run_profile 128 8 1 true 16 "lora_r16"
run_profile 128 8 1 true 32 "lora_r32"

# Test with larger physical batch sizes (if memory allows)
echo "=== Testing Larger Physical Batch Sizes ==="
run_profile 128 8 2 false 16 "physical_batch_2"
run_profile 128 8 4 false 16 "physical_batch_4"

echo "=== Analysis Complete ==="
echo "Results saved in: $RESULTS_DIR"
echo ""
echo "Summary of configurations tested:"
echo "1. Basic configuration (seq=128, batch=8, no LoRA)"
echo "2. Different sequence lengths (64, 256)"
echo "3. Different batch sizes (16, 32)"
echo "4. LoRA configurations (rank 16, 32)"
echo "5. Different physical batch sizes (2, 4)"
echo ""
echo "Check individual result directories for detailed analysis and visualizations."