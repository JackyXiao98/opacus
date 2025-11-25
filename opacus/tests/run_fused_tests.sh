#!/bin/bash
# Run fused flash linear FSDP tests

echo "========================================================================"
echo "Running Fused Flash Linear FSDP Tests"
echo "========================================================================"
echo ""

# Check if in virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Test Categories:"
echo "1. Kernel Tests (CPU-safe)"
echo "2. Module Tests (CPU-safe)"
echo "3. Integration Tests (CPU-safe)"
echo "4. Triton Tests (GPU-only)"
echo "5. Full Suite"
echo ""

# Default: run all CPU-safe tests
TEST_PATTERN="test_fused_flash_linear_fsdp.py"

if [ "$1" == "triton" ]; then
    echo "Running Triton kernel tests (requires GPU)..."
    TEST_PATTERN="test_fused_flash_linear_fsdp.TestTritonFusedKernel"
elif [ "$1" == "kernel" ]; then
    echo "Running kernel correctness tests..."
    TEST_PATTERN="test_fused_flash_linear_fsdp.TestFusedFlashLinearKernels"
elif [ "$1" == "module" ]; then
    echo "Running module tests..."
    TEST_PATTERN="test_fused_flash_linear_fsdp.TestFusedFlashLinearModule"
elif [ "$1" == "integration" ]; then
    echo "Running integration tests..."
    TEST_PATTERN="test_fused_flash_linear_fsdp.TestGradSampleModuleFSDPFuse"
elif [ "$1" == "performance" ]; then
    echo "Running performance comparison tests..."
    TEST_PATTERN="test_fused_flash_linear_fsdp.TestPerformanceComparison"
elif [ "$1" == "all" ]; then
    echo "Running full test suite..."
    TEST_PATTERN="test_fused_flash_linear_fsdp.py"
else
    echo "Running CPU-safe tests (excluding GPU-only Triton tests)..."
    echo ""
fi

echo "========================================================================"
echo ""

# Run tests with verbose output
python -m pytest opacus/tests/$TEST_PATTERN -v -s

EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
else
    echo -e "${RED}❌ Some tests failed (exit code: $EXIT_CODE)${NC}"
fi
echo "========================================================================"

exit $EXIT_CODE

