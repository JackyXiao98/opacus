# FSDP Flash Clipping Profiling and Debugging Guide

## Overview

This guide explains how to use the profiling and correctness verification tools to identify bottlenecks in FSDP Flash Clipping with long sequences.

## Changes Made

### 1. Bug Fixes in `opacus/grad_sample/triton_kernels.py`

**Fixed Issues:**
- **Line 829**: Removed dead code `ret[layer.weight] = torch.ones(B, device=A.device, dtype=dtype_acc)` that was immediately overwritten
- **Line 847**: Removed dead code `ret[layer.bias] = torch.ones(B, device=A.device, dtype=dtype_acc)+0.1` that was immediately overwritten  
- **Line 799, 858-859**: Removed debug timing code (`time_start`, `time_end`, print statement)

**Impact:** These bugs would cause incorrect gradient norm computation for bias parameters in 3D cases, leading to correctness failures in multi-GPU setups.

### 2. Profiling Hooks Added

#### In `opacus/grad_sample/grad_sample_module_fast_gradient_clipping_fsdp.py`:
- Added timing hooks in `get_norm_sample()` to track:
  - Norm stacking time
  - Local computation time
  - **All-reduce communication time** (key bottleneck)
  - Final computation time
- Added per-layer norm computation timing in `capture_backprops_hook()`
- Logs pre/post all-reduce norm values for correctness debugging

#### In `opacus/utils/fast_gradient_clipping_utils.py`:
- Added timing for first backward pass (norm computation)
- Added timing for second backward pass (gradient computation)
- Added timing for bookkeeping mode
- Tracks clipping coefficient computation time

**Enable Profiling:**
```bash
export OPACUS_PROFILE_FSDP=1
```

### 3. New Tools Created

#### `profile_fsdp_bottlenecks.py`
Comprehensive profiling script that:
- Tests multiple sequence lengths (1K, 2K, 4K, 8K, 16K)
- Measures forward/backward/total time
- Tracks CUDA memory usage
- Analyzes scaling behavior (O(T) vs O(T²))
- Supports both Flash Clipping and Bookkeeping modes

#### `verify_correctness.py` (Enhanced)
Added distributed testing support:
- FSDP wrapper option (`--use-fsdp`)
- Distributed launch compatibility
- Verbose logging for debugging (`--verbose`)
- Profiling mode (`--enable-profiling`)

## Running the Tests

### 1. Single-GPU Correctness Test (CPU)

```bash
cd /Users/bytedance/Desktop/Github/opacus
source .venv/bin/activate
python memory_test/fastdp_bookkeeping/verify_correctness.py
```

### 2. Single-GPU Profiling (CPU - for syntax validation)

```bash
python memory_test/fastdp_bookkeeping/profile_fsdp_bottlenecks.py \
    --seq-lengths 512 1024 2048 \
    --batch-size 4 \
    --cpu
```

### 3. Multi-GPU Correctness Test (Requires 2 GPUs)

```bash
# With torchrun (PyTorch 1.9+)
torchrun --nproc_per_node=2 \
    memory_test/fastdp_bookkeeping/verify_correctness.py \
    --use-fsdp \
    --verbose \
    --enable-profiling

# Or with torch.distributed.launch (older PyTorch)
python -m torch.distributed.launch --nproc_per_node=2 \
    memory_test/fastdp_bookkeeping/verify_correctness.py \
    --use-fsdp \
    --verbose \
    --enable-profiling
```

### 4. Multi-GPU Profiling (Requires 2 GPUs)

```bash
export OPACUS_PROFILE_FSDP=1

torchrun --nproc_per_node=2 \
    memory_test/fastdp_bookkeeping/profile_fsdp_bottlenecks.py \
    --seq-lengths 1024 2048 4096 8192 16384 \
    --batch-size 4 \
    --use-flash
```

## Expected Bottlenecks

Based on the code analysis, the likely bottlenecks are:

### 1. **Blocking All-Reduce (Most Likely)**
- **Location:** `grad_sample_module_fast_gradient_clipping_fsdp.py:154` 
- **Issue:** All-reduce happens AFTER all layers finish computing norms
- **Impact:** Prevents overlapping computation with communication
- **Detection:** High all-reduce time in profiling output, low GPU utilization

### 2. **Sequential Layer Processing**
- **Issue:** Each layer computes norms sequentially without freeing activations
- **Impact:** High memory usage, no parallelism across layers
- **Detection:** Memory steadily increases during backward pass

### 3. **Triton Kernel Overhead**  
- **Issue:** For long sequences, triton kernel launch overhead may dominate
- **Impact:** O(T²) complexity in some algorithms
- **Detection:** Per-layer timing shows disproportionate time in norm computation

## Correctness Debugging

If 2-GPU tests fail, enable profiling to see:

```bash
export OPACUS_PROFILE_FSDP=1
torchrun --nproc_per_node=2 \
    memory_test/fastdp_bookkeeping/verify_correctness.py \
    --use-fsdp \
    --verbose \
    --enable-profiling
```

**Look for:**
1. Pre-allreduce squared norms (should be different on each rank)
2. Post-allreduce squared norms (should be identical on all ranks)
3. Final gradient norms (should match single-GPU reference)

**Common Issues:**
- If pre-allreduce norms are already identical → layers not being sharded correctly
- If post-allreduce norms differ → all-reduce not synchronizing properly
- If bias norms are wrong → check triton_kernels.py fixes were applied

## Optimization Strategies

Based on profiling results:

### If Bottleneck is All-Reduce:
1. **Make all-reduce async:** Launch all-reduce as soon as local norms are computed
2. **Overlap with next layer:** Start next layer's computation while all-reduce runs
3. **Use NCCL groups:** Create separate process groups for norm all-reduce

### If Bottleneck is Memory:
1. **Eager activation cleanup:** Delete activations immediately after norm computation
2. **Streaming norm computation:** Process layers one-by-one with immediate cleanup
3. **Activation checkpointing:** Trade compute for memory

### If Bottleneck is Triton Kernels:
1. **Algorithm selection:** Switch between "input_length" and "width" based on T
2. **Tile size tuning:** Adjust tile_size parameter for better memory access
3. **Kernel fusion:** Combine multiple operations into single kernel

## Expected Results

### Correctness
- All tests should pass on 1 GPU and 2 GPUs
- Max gradient difference should be < 1e-5
- Bias norms should be computed correctly (not hardcoded)

### Performance Scaling
For Flash Clipping with long sequences:
- **Forward pass:** O(T) - linear scaling
- **Backward pass (ideal):** O(T) - with optimizations
- **Backward pass (current):** May show O(T²) for very long sequences
- **All-reduce time:** Should be constant (only communicating per-sample norms, not gradients)

### Memory Usage
- Memory should scale linearly with sequence length
- Bookkeeping mode should use slightly more memory (caches activations)
- Peak memory after cleanup should be reasonable

## Next Steps

1. **Run profiling:** Execute profiling script on GPU system
2. **Analyze results:** Identify primary bottleneck from timing breakdown
3. **Implement optimization:** Target the identified bottleneck
4. **Verify correctness:** Ensure optimization doesn't break correctness
5. **Benchmark improvement:** Compare before/after performance

## Files Modified

- `opacus/grad_sample/triton_kernels.py` - Bug fixes
- `opacus/grad_sample/grad_sample_module_fast_gradient_clipping_fsdp.py` - Profiling hooks
- `opacus/utils/fast_gradient_clipping_utils.py` - Profiling hooks
- `memory_test/fastdp_bookkeeping/profile_fsdp_bottlenecks.py` - New profiling script
- `memory_test/fastdp_bookkeeping/verify_correctness.py` - Distributed support

## Contact

For questions or issues, refer to the plan document at `/f.plan.md`.

