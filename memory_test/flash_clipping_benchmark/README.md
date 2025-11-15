# Flash Clipping Algorithm Benchmark

This directory contains a comprehensive benchmarking suite for comparing different gradient norm computation algorithms used in differential privacy training.

## Overview

When computing per-sample gradient norms for differential privacy (DP-SGD), we need efficient algorithms that avoid materializing full per-sample gradients. This benchmark compares four approaches:

1. **PyTorch Input-Length Algorithm** - O(T × d²) complexity, optimal for long sequences
2. **PyTorch Width Algorithm** - O(T² × d) complexity, optimal for wide models
3. **Triton Input-Length Algorithm** - Custom Triton kernels for input-length approach
4. **Triton Width Algorithm** - Custom Triton kernels for width approach

## Algorithms Explained

### Input-Length-Linear Algorithm

**Complexity**: O(T × d²) where T is sequence length, d is feature dimension

**Strategy**:
- Tiles the sequence into blocks of size B_T
- Pre-computes M_j = A_j^T @ G_j for each block
- Computes Frobenius inner products between block pairs
- Avoids materializing full T×T score matrices

**Best for**: Long sequences (large T), moderate dimensions

### Width-Linear Algorithm

**Complexity**: O(T² × d) where T is sequence length, d is feature dimension

**Strategy**:
- Tiles the sequence into blocks of size B_T
- Computes score matrices Score_a = A_j @ A_k^T and Score_g = G_j @ G_k^T
- Performs element-wise multiplication and sum
- Processes smaller T×T blocks instead of full d×d matrices

**Best for**: Wide models (large d), shorter sequences

### Triton Acceleration

The Triton versions use custom GPU kernels that:
- Fuse multiple operations to reduce memory traffic
- Avoid materializing intermediate results in global memory
- Optimize memory access patterns for GPU architecture
- May provide speedups over PyTorch's cuBLAS for certain shapes

## Files

- `benchmark_algorithms.py` - Main benchmarking script
- `visualize_results.py` - Generate plots and tables from results
- `run_benchmark.sh` - Convenience script to run full pipeline
- `README.md` - This file

## Requirements

```bash
# Core requirements
torch>=2.0.0
matplotlib>=3.5.0

# Optional (for Triton acceleration)
triton>=2.0.0
```

## Usage

### Quick Start

Run the full benchmark suite with visualization:

```bash
./run_benchmark.sh
```

For a quick test with smaller tensors:

```bash
./run_benchmark.sh --small
```

For CPU-only testing (no GPU required):

```bash
./run_benchmark.sh --cpu
```

### Manual Usage

#### 1. Run Benchmarks

```bash
# Full benchmark on GPU
python benchmark_algorithms.py --device cuda --output benchmark_results.json

# Small test
python benchmark_algorithms.py --device cuda --output test_results.json --small

# CPU only
python benchmark_algorithms.py --device cpu --output cpu_results.json
```

#### 2. Generate Visualizations

```bash
python visualize_results.py benchmark_results.json --output-dir plots
```

This generates:
- Time comparison bar charts
- Speedup comparison charts
- Memory usage comparison
- Tile size impact analysis
- Summary tables (text files)
- Best configuration recommendations

## Output Files

After running the benchmark, you'll get:

### JSON Results
- `benchmark_results.json` - Raw benchmark data

### Plots Directory
- `time_comparison_*.png` - Time comparison for each configuration
- `speedup_comparison_*.png` - Speedup relative to slowest algorithm
- `memory_comparison_*.png` - Peak memory usage comparison
- `tile_analysis_*.png` - Impact of tile size on performance
- `summary_table.txt` - Comprehensive results table
- `best_configs.txt` - Recommended configurations

## Test Configurations

### Default Shapes (based on user's workload)

1. **Shape 1**: A=[2, 16384, 2048], G=[2, 16384, 512]
   - 2 samples, 16K sequence length, 2048→512 linear layer
   
2. **Shape 2**: A=[2, 16384, 2048], G=[2, 16384, 2048]
   - 2 samples, 16K sequence length, 2048→2048 linear layer

### Tile Sizes Tested
- 256
- 512
- 1024
- 2048

## Interpreting Results

### Time Comparison
- Lower is better
- Look for the algorithm with minimum median time
- Check standard deviation for consistency

### Speedup
- Shows relative performance vs. slowest algorithm
- Values > 1.0 indicate speedup
- Higher is better

### Memory Usage
- Peak memory allocated during computation
- Lower is better for memory-constrained scenarios
- Trade-off with speed may exist

### Relative Error
- Numerical accuracy vs. baseline (first algorithm)
- Should be < 1e-5 for correct implementations
- Larger errors indicate potential numerical issues

### Tile Size Impact
- Shows optimal tile size for each algorithm
- Too small: overhead from loop iterations
- Too large: cache misses, register spills
- Sweet spot varies by hardware and problem size

## Example Results

```
Shape: A=(2, 16384, 2048), G=(2, 16384, 512), Tile=1024
--------------------------------------------------------------------------------
  Algorithm                     Time (ms)   Speedup  Memory (MB)   Rel Error
  --------------------------------------------------------------------------------
  triton_input_length              15.23      1.00x       245.2      0.00e+00
  pytorch_input_length             18.56      0.82x       312.8      3.24e-07
  triton_width                     42.31      0.36x       187.4      2.15e-07
  pytorch_width                    45.67      0.33x       256.1      2.15e-07
```

In this example:
- Triton input-length is fastest (15.23ms)
- PyTorch input-length is 22% slower
- Width algorithms are ~3x slower for this shape
- All algorithms produce numerically accurate results

## Customization

### Add New Shapes

Edit `benchmark_algorithms.py`:

```python
shapes = [
    ((batch, seq_len, d_in), (batch, seq_len, d_out)),
    # Add your custom shapes here
]
```

### Add New Tile Sizes

```python
tile_sizes = [128, 256, 512, 1024, 2048, 4096]
```

### Modify Number of Runs

```python
benchmark_algorithm(
    algo_fn, A, G, tile_size,
    num_warmup=5,    # Increase for more stable results
    num_runs=20      # Increase for better statistics
)
```

## Troubleshooting

### CUDA Out of Memory

- Use `--small` flag for smaller test
- Reduce batch size or sequence length
- Try smaller tile sizes
- Close other GPU applications

### Triton Not Available

- Install with: `pip install triton`
- Check CUDA version compatibility
- Triton benchmarks will fall back to PyTorch if unavailable

### Slow CPU Performance

- CPU testing is for syntax validation only
- Use `--small` flag for faster CPU tests
- Full benchmarks require GPU for reasonable runtime

### Import Errors

Make sure you're in the virtual environment:

```bash
source ../../.venv/bin/activate
```

Or install requirements:

```bash
cd ../..  # Go to project root
uv pip install torch matplotlib
uv pip install triton  # Optional
```

## Performance Tips

1. **For Long Sequences (T >> d)**:
   - Use input_length algorithm
   - Try Triton version for potential speedup
   - Larger tile sizes may help (1024-2048)

2. **For Wide Models (d >> T)**:
   - Use width algorithm
   - Smaller tile sizes may be better (256-512)
   - PyTorch bmm is highly optimized

3. **Memory Constrained**:
   - Use smaller tile sizes
   - Width algorithm typically uses less memory
   - Triton versions may save memory

4. **Maximum Speed**:
   - Run full benchmark to find optimal config
   - Optimal choice depends on hardware and shape
   - Cache results for production use

## Citation

This benchmark suite evaluates algorithms from:
- Flash Clipping for differential privacy (research implementation)
- PyTorch native implementations
- Custom Triton GPU kernels

## Contact

For questions or issues with the benchmark suite, please refer to the main Opacus repository.

