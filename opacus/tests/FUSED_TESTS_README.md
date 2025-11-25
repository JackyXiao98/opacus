# Fused Flash Linear FSDP Tests

## Overview

Comprehensive test suite for the fused flash linear FSDP implementation, including tests for both `flash_fsdp_fuse` and `flash_fsdp_fuse_bk` modes.

## Test Structure

### 1. TestFusedFlashLinearKernels
Low-level kernel correctness tests.
- `test_input_length_frobenius_2d`: 2D input handling
- `test_input_length_frobenius_3d`: 3D sequence input handling
- `test_width_frobenius_matches_input_length`: Algorithm consistency

### 2. TestFusedFlashLinearModule
FusedFlashLinear module behavior tests.
- `test_forward_matches_linear`: Forward pass equivalence to nn.Linear
- `test_backward_computes_norms`: Norm accumulation in backward pass

### 3. TestReplaceLinearWithFused
Module replacement utility tests.
- `test_replace_preserves_weights`: Weight preservation during replacement
- `test_get_fused_modules`: Module discovery utility

### 4. TestTritonFusedKernel ⚡ **GPU-ONLY**
Triton kernel correctness tests for `flash_fsdp_fuse_bk`.
- `test_triton_kernel_basic`: Basic gradient bookkeeping correctness
- `test_triton_kernel_with_clipping`: Clipping coefficient handling
- `test_triton_kernel_bias_handling`: Bias gradient accumulation

### 5. TestGradSampleModuleFSDPFuse
Full integration tests.
- `test_gsm_class_registration`: Mode registration verification
- `test_wrap_model_fuse`: flash_fsdp_fuse wrapping
- `test_wrap_model_fuse_bk`: flash_fsdp_fuse_bk wrapping
- `test_forward_backward_norms`: Norm computation in training loop
- `test_clipping_coef`: Clipping coefficient computation

### 6. TestPerformanceComparison
Correctness and performance comparison tests.
- `test_fuse_faster_than_hooks_long_sequence`: Performance benchmark
- `test_fused_norms_correct_single_linear`: Single layer norm verification
- `test_fused_norms_correct_multi_linear`: Multi-layer norm verification
- `test_fuse_vs_fuse_bk_consistency`: **flash_fsdp_fuse vs flash_fsdp_fuse_bk**
- `test_fuse_bk_gradient_accumulation_correctness`: Bookkeeping gradient accuracy
- `test_fuse_bk_multiple_iterations`: Multi-iteration stability
- `test_fuse_bk_with_hooks_comparison`: Comparison with hook-based approach
- `test_fuse_bk_loss_reduction_modes`: Different loss reduction modes
- `test_all_modes_consistency`: **Comprehensive mode comparison**

## Running Tests

### Quick Start

```bash
# Run all CPU-safe tests (recommended for CI)
cd opacus/tests
./run_fused_tests.sh

# Or use pytest directly
python -m pytest test_fused_flash_linear_fsdp.py -v
```

### Specific Test Categories

```bash
# Kernel tests only
./run_fused_tests.sh kernel

# Module tests only
./run_fused_tests.sh module

# Integration tests only
./run_fused_tests.sh integration

# Performance comparison tests only
./run_fused_tests.sh performance

# Triton kernel tests (requires GPU)
./run_fused_tests.sh triton

# Full suite (including GPU tests)
./run_fused_tests.sh all
```

### Individual Tests

```bash
# Run a specific test
python -m pytest test_fused_flash_linear_fsdp.py::TestPerformanceComparison::test_fuse_vs_fuse_bk_consistency -v

# Run with output
python -m pytest test_fused_flash_linear_fsdp.py::TestPerformanceComparison::test_all_modes_consistency -v -s
```

## Key Tests for flash_fsdp_fuse_bk

### Correctness Tests

1. **Triton Kernel Correctness** (GPU-only)
   ```bash
   python -m pytest test_fused_flash_linear_fsdp.py::TestTritonFusedKernel -v
   ```
   - Verifies triton fused gradient bookkeeping kernel
   - Tests clipping coefficient application
   - Tests bias gradient handling

2. **Mode Consistency**
   ```bash
   python -m pytest test_fused_flash_linear_fsdp.py::TestPerformanceComparison::test_fuse_vs_fuse_bk_consistency -v
   ```
   - Compares flash_fsdp_fuse and flash_fsdp_fuse_bk
   - Should produce identical norms (within tolerance)

3. **Gradient Accumulation**
   ```bash
   python -m pytest test_fused_flash_linear_fsdp.py::TestPerformanceComparison::test_fuse_bk_gradient_accumulation_correctness -v
   ```
   - Verifies clipped gradients are correctly accumulated
   - Manual computation vs kernel comparison

4. **Comprehensive Comparison**
   ```bash
   python -m pytest test_fused_flash_linear_fsdp.py::TestPerformanceComparison::test_all_modes_consistency -v -s
   ```
   - Compares flash, flash_fsdp_fuse, and flash_fsdp_fuse_bk
   - All modes should produce consistent results

### Integration Tests

```bash
# Test multiple training iterations
python -m pytest test_fused_flash_linear_fsdp.py::TestPerformanceComparison::test_fuse_bk_multiple_iterations -v

# Test different loss reduction modes
python -m pytest test_fused_flash_linear_fsdp.py::TestPerformanceComparison::test_fuse_bk_loss_reduction_modes -v

# Compare with hook-based approach
python -m pytest test_fused_flash_linear_fsdp.py::TestPerformanceComparison::test_fuse_bk_with_hooks_comparison -v
```

## Expected Results

### Correctness Criteria

1. **Forward Pass**: All modes produce identical outputs
2. **Norm Computation**: 
   - flash_fsdp_fuse and flash_fsdp_fuse_bk: < 0.1% difference
   - flash_fsdp_fuse_bk vs hook-based: < 1% difference (algorithmic differences)
3. **Gradient Accumulation**: < 0.1% relative error for flash_fsdp_fuse_bk

### Tolerance Levels

```python
# Output matching
rtol=1e-5, atol=1e-6  # Very strict

# Norm matching (fuse vs fuse_bk)
rtol=1e-3, atol=1e-4  # Strict

# Norm matching (fuse_bk vs hooks)
rtol=1e-2, atol=1e-3  # Relaxed (algorithmic differences)
```

## Troubleshooting

### Triton Tests Fail

**Issue**: Triton kernel tests fail or are skipped
**Solution**: 
- Triton tests require GPU (CUDA)
- Run on CPU: Triton tests will be skipped automatically
- Install triton: `pip install triton` (if not installed)

### Norm Differences Too Large

**Issue**: Norm differences exceed tolerance
**Solution**:
- Check random seeds are consistent
- Verify model initialization is identical
- Ensure same batch/sequence dimensions
- Check for numerical instability (e.g., very large/small values)

### Import Errors

**Issue**: Cannot import fused modules
**Solution**:
```bash
# Ensure opacus is in PYTHONPATH
export PYTHONPATH=/path/to/opacus:$PYTHONPATH

# Or install in development mode
pip install -e .
```

## Performance Notes

### CPU vs GPU

- **CPU**: Triton tests are skipped, fused approach may not show speedup
- **GPU**: Triton kernel provides significant speedup with FSDP
- **GPU + FSDP**: Maximum benefit from reduced hook overhead

### Expected Speedups (GPU + FSDP)

- flash_fsdp_fuse vs flash: 1.5-2x faster (eliminates hook overhead)
- flash_fsdp_fuse_bk vs flash_fsdp_fuse: Similar or slightly faster (triton kernel)
- flash_fsdp_fuse_bk vs flash: 1.5-2.5x faster (combined benefits)

## CI/CD Integration

### Recommended CI Configuration

```yaml
# Run CPU-safe tests in CI
- name: Test Fused Flash Linear
  run: |
    python -m pytest opacus/tests/test_fused_flash_linear_fsdp.py \
      -v --ignore-glob="*TestTritonFusedKernel*"

# GPU tests (if GPU available)
- name: Test Triton Kernels
  if: ${{ matrix.gpu == 'enabled' }}
  run: |
    python -m pytest opacus/tests/test_fused_flash_linear_fsdp.py::TestTritonFusedKernel -v
```

## References

- [Fused Flash Linear Implementation](../opacus/grad_sample/fused_flash_linear.py)
- [FSDP Fuse Module](../opacus/grad_sample/grad_sample_module_fast_gradient_clipping_fsdp_fuse.py)
- [Triton Kernel](../opacus/grad_sample/triton_fused_kernel.py)
- [Optimization Notes](../../FUSED_FLASH_LINEAR_OPTIMIZATION.md)

