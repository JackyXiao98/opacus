#!/bin/bash

# GPU Memory Analysis Script for Opacus
# Usage: ./run_memory_analysis.sh [basic|detailed] [HF_TOKEN]

set -e

# Default parameters
ANALYSIS_TYPE=${1:-"basic"}
HF_TOKEN=${2:-""}
RESULTS_DIR="./memory_analysis_$(date +%Y%m%d_%H%M%S)"

# Check if token is provided
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HuggingFace token is required"
    echo "Usage: $0 [basic|detailed] [HF_TOKEN]"
    exit 1
fi

# Check CUDA availability
if ! python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')"; then
    echo "Error: CUDA not available or PyTorch not installed"
    exit 1
fi

echo "Starting GPU Memory Analysis..."
echo "Analysis type: $ANALYSIS_TYPE"
echo "Results directory: $RESULTS_DIR"

# Create results directory
mkdir -p "$RESULTS_DIR"

if [ "$ANALYSIS_TYPE" = "basic" ]; then
    echo "Running basic memory analysis..."
    python gpu_memory_profiler.py \
        --token "$HF_TOKEN" \
        --seq_lengths 128 256 512 \
        --batch_sizes 8 16 32 \
        --max_physical_batch_sizes 1 2 4 \
        --lora_ranks 8 16 \
        --results_dir "$RESULTS_DIR" \
        --num_steps 5 \
        --model_name "meta-llama/Llama-3.1-8B-Instruct"

elif [ "$ANALYSIS_TYPE" = "detailed" ]; then
    echo "Running detailed memory analysis..."
    python detailed_memory_profiler.py \
        --token "$HF_TOKEN" \
        --seq_lengths 128 256 \
        --batch_sizes 8 16 \
        --max_physical_batch_sizes 1 2 \
        --results_dir "$RESULTS_DIR" \
        --num_steps 3 \
        --model_name "meta-llama/Llama-3.1-8B-Instruct"

else
    echo "Error: Invalid analysis type. Use 'basic' or 'detailed'"
    exit 1
fi

echo "Analysis completed successfully!"
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Generated files:"
if [ "$ANALYSIS_TYPE" = "basic" ]; then
    echo "  - memory_profile_results.json"
    echo "  - memory_profile_summary.csv"
    echo "  - memory_analysis_overview.png"
    echo "  - memory_timeline_config_*.png"
    echo "  - memory_heatmap.png"
else
    echo "  - detailed_memory_results.json"
    echo "  - detailed_memory_analysis.png"
    echo "  - operation_level_breakdown.png"
fi

echo ""
echo "To view results:"
echo "  cd $RESULTS_DIR"
echo "  ls -la"