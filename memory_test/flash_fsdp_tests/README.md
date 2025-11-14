# Flash Clipping FSDP Tests

This directory contains comprehensive tests for Flash Clipping with FSDP support.

## Overview

Flash Clipping is an accelerated implementation of Ghost Clipping that uses optimized kernels (via Triton) for computing per-sample gradient norms. This test suite validates that Flash Clipping works correctly with FSDP (Fully Sharded Data Parallel) and produces numerically accurate results compared to single GPU training.

## Files

- `test_model.py`: Small transformer model (~10M params) with DPMultiheadAttentionWithFlashAttention
- `test_single_gpu.py`: Single GPU baseline training with `grad_sample_mode="flash"`
- `test_fsdp_multi_gpu.py`: FSDP training with `grad_sample_mode="flash_fsdp"`
- `test_accuracy_comparison.py`: Automated comparison test with strict numerical tolerance
- `README.md`: This file

## Quick Start

### Prerequisites

```bash
# Activate virtual environment
source .venv/bin/activate

# Ensure opacus is installed in development mode
cd /Users/bytedance/Desktop/Github/opacus
pip install -e .
```

### Run Single GPU Training

```bash
cd memory_test/flash_fsdp_tests
python test_single_gpu.py \
    --batch_size 32 \
    --num_samples 1000 \
    --seq_len 64 \
    --num_epochs 1 \
    --lr 1e-3 \
    --noise_multiplier 1.0 \
    --max_grad_norm 1.0 \
    --seed 42
```

### Run FSDP Training

```bash
# Single GPU FSDP (for comparison)
python test_fsdp_multi_gpu.py \
    --batch_size 32 \
    --num_samples 1000 \
    --seq_len 64 \
    --num_epochs 1 \
    --lr 1e-3 \
    --noise_multiplier 1.0 \
    --max_grad_norm 1.0 \
    --seed 42 \
    --world_size 1

# Multi-GPU FSDP (if available)
python test_fsdp_multi_gpu.py \
    --batch_size 32 \
    --num_samples 1000 \
    --seq_len 64 \
    --num_epochs 1 \
    --lr 1e-3 \
    --noise_multiplier 1.0 \
    --max_grad_norm 1.0 \
    --seed 42
```

### Run Accuracy Comparison

```bash
python test_accuracy_comparison.py \
    --batch_size 32 \
    --num_samples 1000 \
    --seq_len 64 \
    --num_epochs 1 \
    --lr 1e-3 \
    --noise_multiplier 1.0 \
    --max_grad_norm 1.0 \
    --seed 42 \
    --tolerance 1e-4
```

This will:
1. Run single GPU training with `grad_sample_mode="flash"`
2. Run FSDP training with `grad_sample_mode="flash_fsdp"`
3. Compare metrics (loss, gradient norms, parameter norms)
4. Generate comparison plots
5. Pass/fail based on numerical tolerance

## Model Architecture

The test model (`SmallTransformerDP`) consists of:
- Token embedding layer
- Positional embedding layer
- 3 transformer blocks with:
  - Multi-head self-attention using Flash Attention
  - Feed-forward network (MLP)
  - Layer normalization
- Classification head

Total parameters: ~10M

## Accuracy Validation

The comparison test validates:

1. **Loss values**: Must match within tolerance (default 1e-4 for FP32)
2. **Gradient norms**: Must match within 10x tolerance
3. **Parameter norms**: Must match within tolerance

The test uses:
- Fixed random seed for reproducibility
- No dropout for deterministic results
- No Poisson sampling for exact batch ordering
- Same hyperparameters across both modes

## Expected Results

When working correctly:
- Loss curves should be nearly identical
- Max loss difference should be < 1e-4 (FP32) or < 1e-3 (BF16)
- Parameter norms should match closely
- Privacy budget (epsilon) should be identical

## Troubleshooting

### Test fails with large differences

Possible causes:
- Different random seeds or non-deterministic operations
- FSDP all-reduce precision issues
- Flash Clipping kernel bugs

Check:
```bash
# Verify seeds are set correctly
grep "torch.manual_seed" test_*.py

# Check for non-deterministic operations
# Look for shuffle=True, dropout>0, etc.
```

### Out of memory errors

Reduce batch size or sequence length:
```bash
python test_accuracy_comparison.py \
    --batch_size 16 \
    --seq_len 32 \
    --num_samples 500
```

### Import errors

Ensure paths are set correctly:
```bash
cd memory_test/flash_fsdp_tests
python -c "from test_model import SmallTransformerDP; print('OK')"
```

## Implementation Notes

### Flash Clipping vs Ghost Clipping

- **Ghost Clipping**: Standard implementation, computes gradient norms on-the-fly
- **Flash Clipping**: Optimized with Triton kernels for sequence models
- Both produce identical results, Flash Clipping is faster for long sequences

### FSDP Integration

The `flash_fsdp` mode:
1. Uses `GradSampleModuleFastGradientClippingFSDP` wrapper
2. Supports `FLASH_NORM_SAMPLERS` in addition to `NORM_SAMPLERS`
3. Handles FSDP-specific module types correctly
4. Performs all-reduce for gradient norms across ranks

## Contributing

When adding new tests:
1. Follow the existing pattern for metrics tracking
2. Use fixed seeds for reproducibility
3. Add both unit tests and integration tests
4. Document expected behavior and tolerances

## References

- FastDP paper: [link if available]
- Ghost Clipping: Opacus documentation
- Flash Attention: Dao et al., 2022
- FSDP: PyTorch documentation

