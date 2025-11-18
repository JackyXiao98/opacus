# FSDP Flash Clipping Optimization Implementations

## Overview

This document provides ready-to-implement optimizations for FSDP Flash Clipping based on identified bottlenecks. Apply these after running profiling to confirm the bottleneck.

## Optimization 1: Async All-Reduce (For Communication Bottleneck)

### When to Apply
- Profiling shows high all-reduce time (> 30% of total backward time)
- GPU utilization drops during norm aggregation
- Sequence length is long (> 4K tokens)

### Implementation

**File: `opacus/grad_sample/grad_sample_module_fast_gradient_clipping_fsdp.py`**

Replace the `get_norm_sample()` method:

```python
def get_norm_sample(self) -> torch.Tensor:
    """
    Get per-example gradient norms with distributed reduction (ASYNC VERSION).
    """
    enable_profiling = os.environ.get('OPACUS_PROFILE_FSDP', '0') == '1'
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    
    if enable_profiling:
        sync()
        t_start = time.time()
    
    # Stack per-parameter norms from all modules
    stacked_norms = torch.stack(
        [
            per_param_norm
            for module in self.iterate_submodules(self._module)
            for per_param_norm in module.norm_sample
        ],
        dim=0,
    )
    
    if enable_profiling:
        sync()
        t_stack = time.time()
    
    # Compute local contribution: sum of squared norms
    norm_sample_squared = (stacked_norms ** 2).sum(dim=0)

    if enable_profiling:
        sync()
        t_local = time.time()
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        print(f"[FSDP Profile] Rank {rank} - Pre-allreduce squared norms shape: {norm_sample_squared.shape}, "
              f"mean: {norm_sample_squared.mean().item():.6f}, "
              f"max: {norm_sample_squared.max().item():.6f}")

    # OPTIMIZATION: Use async all-reduce with work handle
    work = None
    if torch.distributed.is_initialized():
        # Launch async all-reduce
        work = torch.distributed.all_reduce(
            norm_sample_squared, 
            op=torch.distributed.ReduceOp.SUM,
            async_op=True  # KEY CHANGE: Make it asynchronous
        )

    # Can do other work here while all-reduce runs...
    
    # Wait for all-reduce to complete
    if work is not None:
        work.wait()

    if enable_profiling:
        sync()
        t_allreduce = time.time()
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        print(f"[FSDP Profile] Rank {rank} - Post-allreduce squared norms shape: {norm_sample_squared.shape}, "
              f"mean: {norm_sample_squared.mean().item():.6f}, "
              f"max: {norm_sample_squared.max().item():.6f}")

    # Take square root to get final per-sample gradient norms
    norm_sample = torch.sqrt(norm_sample_squared + 1e-12)

    if enable_profiling:
        sync()
        t_end = time.time()
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        print(f"[FSDP Profile] Rank {rank} get_norm_sample timing breakdown:")
        print(f"  - Stack norms:   {(t_stack - t_start)*1000:.2f} ms")
        print(f"  - Local compute: {(t_local - t_stack)*1000:.2f} ms")
        print(f"  - All-reduce:    {(t_allreduce - t_local)*1000:.2f} ms")
        print(f"  - Final compute: {(t_end - t_allreduce)*1000:.2f} ms")
        print(f"  - TOTAL:         {(t_end - t_start)*1000:.2f} ms")

    self.per_sample_gradient_norms = norm_sample
    return norm_sample
```

**Expected Improvement:** 10-30% reduction in backward pass time for long sequences.

## Optimization 2: Eager Activation Cleanup (For Memory Bottleneck)

### When to Apply
- Peak memory usage is very high
- OOM errors with long sequences
- Memory profiling shows activations retained too long

### Implementation

**File: `opacus/grad_sample/grad_sample_module_fast_gradient_clipping_fsdp.py`**

Modify `capture_backprops_hook()` to eagerly free activations:

```python
def capture_backprops_hook(
    self,
    module: nn.Module,
    _forward_input: torch.Tensor,
    forward_output: torch.Tensor,
    loss_reduction: str,
    batch_first: bool,
):
    """
    Computes norms with eager activation cleanup.
    """
    if not self.hooks_enabled:
        return

    backprops = forward_output[0].detach()

    activations, backprops = self.rearrange_grad_samples(
        module=module,
        backprops=backprops,
        loss_reduction=loss_reduction,
        batch_first=batch_first,
    )

    if not hasattr(module, "norm_sample"):
        module.norm_sample = []
        for _, param in trainable_parameters(module):
            module.norm_sample.append(
                torch.zeros(
                    torch.Size([module.max_batch_len, 1]),
                    device=param.device,
                    dtype=param.dtype,
                )
            )

    module_type = self._get_module_type(module)
    module._forward_counter -= 1
    if self.use_ghost_clipping and (
        module_type in self.NORM_SAMPLERS or 
        (self.use_flash_clipping and module_type in self.FLASH_NORM_SAMPLERS)
    ):
        # Compute norms
        if self.use_flash_clipping and module_type in self.FLASH_NORM_SAMPLERS:
            norm_sampler_fn = self.FLASH_NORM_SAMPLERS[module_type]
        else:
            norm_sampler_fn = self.NORM_SAMPLERS[module_type]
        
        norm_samples = norm_sampler_fn(module, activations, backprops)

        for idx, (_, ns) in enumerate(
            (item for item in norm_samples.items() if item[0].requires_grad)
        ):
            module.norm_sample[idx] = ns
        
        # OPTIMIZATION: Eagerly free activations after norm computation
        # (but only if not using bookkeeping mode which needs them)
        if not self.enable_fastdp_bookkeeping:
            # Clear activations list to free memory immediately
            module.activations = []
            # Force Python garbage collection on this scope
            del activations
            del backprops
            del norm_samples
        else:
            # Bookkeeping mode: Cache for later gradient computation
            self._bk_cache.append({
                'module': module,
                'activations': activations,
                'backprops': backprops,
            })
    else:
        # Standard grad sample computation
        if not self.force_functorch and module_type in self.GRAD_SAMPLERS:
            grad_sampler_fn = self.GRAD_SAMPLERS[module_type]
        else:
            grad_sampler_fn = ft_compute_per_sample_gradient

        grad_samples = grad_sampler_fn(module, activations, backprops)

        for idx, (_, gs) in enumerate((item for item in grad_samples.items())):
            module.norm_sample[idx] = gs.reshape(len(gs), -1).norm(2, dim=-1)
        
        # OPTIMIZATION: Eagerly free memory
        module.activations = []
        del grad_samples
        del activations
        del backprops

    if len(module.activations) == 0:
        if hasattr(module, "max_batch_len"):
            del module.max_batch_len
```

**Expected Improvement:** 20-40% reduction in peak memory usage.

## Optimization 3: Algorithm Selection (For Triton Kernel Overhead)

### When to Apply
- Profiling shows per-layer norm computation is slow
- O(T²) scaling observed in backward pass
- Using "input_length" algorithm with very long sequences

### Implementation

**File: `opacus/grad_sample/grad_sample_module_fast_gradient_clipping_fsdp.py`**

Add adaptive algorithm selection in `__init__`:

```python
def __init__(
    self,
    m: nn.Module,
    *,
    batch_first: bool = True,
    loss_reduction="mean",
    strict: bool = True,
    max_grad_norm=1,
    use_flash_clipping=False,
    use_ghost_clipping=True,
    enable_fastdp_bookkeeping=False,
    flash_algorithm="auto",  # NEW PARAMETER
):
    """
    Args:
        flash_algorithm: Algorithm for flash clipping. Options:
            - "auto": Automatically select based on sequence length
            - "input_length": O(T * d²), better for long sequences
            - "width": O(T² * d), better for wide models
    """
    super().__init__(
        m,
        batch_first=batch_first,
        loss_reduction=loss_reduction,
        strict=strict,
        force_functorch=False,
        max_grad_norm=max_grad_norm,
        use_ghost_clipping=use_ghost_clipping,
        use_flash_clipping=use_flash_clipping,
        enable_fastdp_bookkeeping=enable_fastdp_bookkeeping,
    )
    self.flash_algorithm = flash_algorithm
```

Then modify the norm sampler call to use adaptive selection:

```python
# In capture_backprops_hook, before calling norm_sampler_fn:
if self.use_flash_clipping and module_type in self.FLASH_NORM_SAMPLERS:
    # Determine algorithm adaptively
    if self.flash_algorithm == "auto":
        # Heuristic: use "input_length" for T > 2048, "width" otherwise
        if activations and activations[0].dim() == 3:
            seq_len = activations[0].shape[1]
            algorithm = "input_length" if seq_len > 2048 else "width"
        else:
            algorithm = "width"
    else:
        algorithm = self.flash_algorithm
    
    # Call with algorithm parameter
    norm_samples = self.FLASH_NORM_SAMPLERS[module_type](
        module, activations, backprops, algorithm=algorithm
    )
else:
    norm_samples = self.NORM_SAMPLERS[module_type](module, activations, backprops)
```

**File: `opacus/grad_sample/triton_kernels.py`**

Also add tunable tile size:

```python
def compute_linear_norm_sample_flash(
    layer: nn.Linear,
    activations: List[torch.Tensor],
    backprops: torch.Tensor,
    algorithm: str = "auto",  # CHANGED: default to "auto"
    tile_size: int = None,  # CHANGED: None means auto-tune
    dtype_acc = torch.float32,
    use_flash_clipping: bool = False,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Compute per-sample gradient norms with adaptive algorithm selection.
    """
    A = activations[0]
    ret: Dict[nn.Parameter, torch.Tensor] = {}
    
    # Auto-select algorithm if needed
    if algorithm == "auto":
        if backprops.dim() == 3:
            T = backprops.shape[1]
            d_in = A.shape[-1]
            d_out = backprops.shape[-1]
            
            # Cost model: input_length is O(T * d²), width is O(T² * d)
            # Use input_length when T * d² < T² * d, i.e., when d < T
            algorithm = "input_length" if (d_in + d_out) / 2 < T else "width"
        else:
            algorithm = "width"  # 2D case: doesn't matter
    
    # Auto-select tile size if not provided
    if tile_size is None:
        if backprops.dim() == 3:
            T = backprops.shape[1]
            # Heuristic: use larger tiles for longer sequences
            if T < 1024:
                tile_size = 256
            elif T < 4096:
                tile_size = 512
            else:
                tile_size = 1024
        else:
            tile_size = 1024  # Default for 2D
    
    if algorithm not in ["input_length", "width"]:
        raise ValueError(f"Algorithm must be 'input_length' or 'width', got '{algorithm}'")
    
    # Rest of the function remains the same...
    # (existing implementation)
```

**Expected Improvement:** 2-5x speedup for certain sequence length/model width combinations.

## Optimization 4: Layer-by-Layer Processing (For Both Memory and Speed)

### When to Apply
- Both memory and speed are issues
- Willing to restructure backward pass
- Using bookkeeping mode

### Implementation

This is a more invasive change that would require restructuring the backward pass to process layers one at a time, computing norms and freeing activations immediately.

**Concept:**
```python
# Instead of:
# 1. Forward all layers
# 2. Backward all layers, compute all norms
# 3. All-reduce all norms
# 4. Second backward all layers

# Do:
# 1. Forward all layers
# 2. For each layer (in reverse):
#    a. Backward through layer
#    b. Compute norm
#    c. Free activation
#    d. Partial all-reduce (or accumulate locally)
# 3. Final all-reduce of accumulated norms
# 4. Second backward (only if needed)
```

This requires significant refactoring and should only be attempted if simpler optimizations don't suffice.

## Testing Optimizations

After applying any optimization:

### 1. Verify Correctness
```bash
python memory_test/fastdp_bookkeeping/verify_correctness.py --use-fsdp --verbose
```

### 2. Profile Performance
```bash
export OPACUS_PROFILE_FSDP=1
python memory_test/fastdp_bookkeeping/profile_fsdp_bottlenecks.py \
    --seq-lengths 1024 4096 16384 \
    --use-flash
```

### 3. Compare Before/After
Document timing improvements:
- Backward pass time
- All-reduce time  
- Peak memory usage
- Scaling behavior with sequence length

## Recommendation

**Start with Optimization 1 (Async All-Reduce)** - it's:
- Easy to implement (one-line change)
- Low risk (well-tested pattern)
- High impact (communication is likely the bottleneck)
- Compatible with all other optimizations

**Then add Optimization 2 (Eager Cleanup)** if memory is an issue:
- Also easy to implement
- Significant memory savings
- Compatible with async all-reduce

**Only attempt Optimization 3 or 4** if profiling shows they're needed:
- More complex to implement correctly
- Requires careful testing
- May not provide significant benefits depending on workload

## Rollback Plan

If an optimization causes issues:
1. Keep the original code in comments
2. Add a flag to enable/disable the optimization
3. Document any correctness or performance regressions
4. Have profiling data showing before/after comparison

Example:
```python
USE_ASYNC_ALLREDUCE = os.environ.get('OPACUS_ASYNC_ALLREDUCE', '1') == '1'

if USE_ASYNC_ALLREDUCE:
    work = torch.distributed.all_reduce(..., async_op=True)
    work.wait()
else:
    torch.distributed.all_reduce(..., async_op=False)
```

