#!/bin/bash
#
# Run Flash Clipping Algorithm Benchmarks
#
# This script runs benchmarks for different gradient norm computation algorithms
# and generates visualization reports.
#
# Usage:
#   ./run_benchmark.sh [--small] [--cpu]
#


# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=================================="
echo "Flash Clipping Algorithm Benchmark"
echo "=================================="
echo ""

# Parse arguments
SMALL_FLAG=""
DEVICE="cuda"

while [[ $# -gt 0 ]]; do
    case $1 in
        --small)
            SMALL_FLAG="--small"
            shift
            ;;
        --cpu)
            DEVICE="cpu"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--small] [--cpu]"
            exit 1
            ;;
    esac
done

# Navigate to project root
cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Warning: Virtual environment not found at .venv"
    echo "Make sure you have the required packages installed."
fi

# Check Python and packages
echo ""
echo "Checking environment..."
python --version
echo ""

# Check if required packages are available
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || {
    echo "Error: PyTorch not found. Please install it first."
    exit 1
}

python -c "import matplotlib; print(f'Matplotlib version: {matplotlib.__version__}')" || {
    echo "Error: Matplotlib not found. Please install it first."
    exit 1
}

# Check CUDA availability
if [ "$DEVICE" = "cuda" ]; then
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || {
        echo "Warning: CUDA not available, falling back to CPU"
        DEVICE="cpu"
    }
fi

# Check Triton availability
echo ""
echo "Checking Triton availability..."
python -c "import triton; print(f'Triton version: {triton.__version__}')" 2>/dev/null || {
    echo "Warning: Triton not available. Triton benchmarks will fall back to PyTorch."
}

echo ""
echo "=================================="
echo "Running Benchmarks"
echo "=================================="
echo ""
echo "Device: $DEVICE"
if [ -n "$SMALL_FLAG" ]; then
    echo "Mode: Small test (for quick validation)"
else
    echo "Mode: Full benchmark"
fi
echo ""

# Run benchmark
cd "$SCRIPT_DIR"

BENCHMARK_OUTPUT="benchmark_results.json"

echo "Starting benchmark..."
echo ""

python benchmark_algorithms.py \
    --device "$DEVICE" \
    --output "$BENCHMARK_OUTPUT" \
    $SMALL_FLAG

echo ""
echo "=================================="
echo "Generating Visualizations"
echo "=================================="
echo ""

# Generate visualizations
python visualize_results.py "$BENCHMARK_OUTPUT" --output-dir plots

echo ""
echo "=================================="
echo "Benchmark Complete!"
echo "=================================="
echo ""
echo "Results saved to:"
echo "  - Benchmark data: $SCRIPT_DIR/$BENCHMARK_OUTPUT"
echo "  - Plots: $SCRIPT_DIR/plots/"
echo ""
echo "Summary files:"
echo "  - $SCRIPT_DIR/plots/summary_table.txt"
echo "  - $SCRIPT_DIR/plots/best_configs.txt"
echo ""
echo "To view the summary:"
echo "  cat $SCRIPT_DIR/plots/summary_table.txt"
echo ""

