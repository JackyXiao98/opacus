#!/bin/bash

echo "=== Installing dependencies for PyTorch Profiling ==="

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.7 or later."
    exit 1
fi

echo "Python version:"
python3 --version

# Check if pip is available
if ! python3 -m pip --version &> /dev/null; then
    echo "Error: pip not found. Please install pip."
    exit 1
fi

echo "Installing PyTorch and dependencies..."

# Try to install from requirements.txt first
if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    python3 -m pip install -r requirements.txt
else
    echo "requirements.txt not found, installing manually..."
    python3 -m pip install torch>=1.12.0 torchvision>=0.13.0
    python3 -m pip install opacus>=1.4.0
    python3 -m pip install tensorboard>=2.8.0
    python3 -m pip install numpy>=1.21.0
fi

echo ""
echo "=== Verifying installation ==="

# Test imports
python3 -c "
try:
    import torch
    print('✓ PyTorch installed successfully:', torch.__version__)
except ImportError:
    print('✗ PyTorch installation failed')

try:
    import opacus
    print('✓ Opacus installed successfully:', opacus.__version__)
except ImportError:
    print('✗ Opacus installation failed')

try:
    import tensorboard
    print('✓ TensorBoard installed successfully')
except ImportError:
    print('✗ TensorBoard installation failed')
"

echo ""
echo "=== Testing script syntax ==="
if [ -f "test_syntax.py" ]; then
    python3 test_syntax.py
else
    echo "test_syntax.py not found, skipping syntax check"
fi

echo ""
echo "=== Testing DP compatibility ==="
if [ -f "test_dp_compatibility.py" ]; then
    python3 test_dp_compatibility.py
else
    echo "test_dp_compatibility.py not found, skipping DP compatibility check"
fi

echo ""
echo "=== Installation complete ==="
echo "You can now run:"
echo "  python3 profiling_script.py --mode=test    # For local testing"
echo "  python3 profiling_script.py --mode=profile # For full profiling"