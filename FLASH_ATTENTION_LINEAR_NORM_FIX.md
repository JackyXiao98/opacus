# Flash Attention Linear Norm Clipping Fix

## Problem Description

When using `SimpleBigModelWithFlashAttention` with Ghost Clipping (`GradSampleModuleFastGradientClipping` with `use_triton=False`), Linear layers were **not** using norm clipping. Instead, no `_norm_sample` attributes were created on Linear layer parameters, meaning gradient norms were not being computed efficiently.

## Root Cause Analysis

The issue was caused by how Opacus's `GradSampleModule` iterates through submodules to register hooks:

### The `iterate_submodules` Logic

```python
def iterate_submodules(self, module: nn.Module) -> Iterable[nn.Module]:
    if has_trainable_params(module):
        yield module

    # Don't recurse if module is handled by functorch
    if (
        has_trainable_params(module)
        and type(module) not in self.GRAD_SAMPLERS
        and type(module) not in [DPRNN, DPLSTM, DPGRU]
    ):
        return  # ← This prevents recursion!

    for m in module.children():
        yield from self.iterate_submodules(m)
```

The key function is `has_trainable_params`:
```python
def has_trainable_params(module: nn.Module) -> bool:
    return any(p.requires_grad for p in module.parameters(recurse=False))
```

Note `recurse=False` - it only checks **direct** parameters owned by the module, not parameters in submodules.

### The Problem

In `SimpleBigModelWithFlashAttention`, the position embedding was defined as:

```python
self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, hidden_dim))
```

This made the top-level `SimpleBigModelWithFlashAttention` module directly own a parameter. When Opacus checked:

1. ✓ `has_trainable_params(SimpleBigModelWithFlashAttention)` → `True` (has direct parameter: pos_embedding)
2. ✓ `type(SimpleBigModelWithFlashAttention) not in GRAD_SAMPLERS` → `True` (custom module, not registered)
3. → Function returns early, **never recursing into child modules** (Linear layers in attention, FFN, etc.)

As a result, hooks were never registered on the Linear layers, and norm sampling never occurred.

### Why SimpleBigModel Worked

In `SimpleBigModel`, the position embedding was defined as:

```python
self.pos_embedding = nn.Embedding(seq_len, hidden_dim)
```

Using `nn.Embedding` means:
- The top-level module has **0 direct parameters** (only child modules have parameters)
- `has_trainable_params(SimpleBigModel)` → `False` for direct parameters
- Opacus **does** recurse into child modules
- Hooks are registered on all Linear layers ✓

## Solution

Changed `SimpleBigModelWithFlashAttention` to use `nn.Embedding` instead of `nn.Parameter`:

### Before (Broken)
```python
class SimpleBigModelWithFlashAttention(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, hidden_dim))
        # ...
    
    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids) + self.pos_embedding
        # ...
```

### After (Fixed)
```python
class SimpleBigModelWithFlashAttention(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        # Use nn.Embedding instead of nn.Parameter to avoid blocking iterate_submodules
        # When the top-level module has direct parameters, Opacus won't recurse into submodules
        self.pos_embedding = nn.Embedding(seq_len, hidden_dim)
        self.seq_len = seq_len
        # ...
    
    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape
        # Create position IDs
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Token embeddings + position embeddings
        x = self.embedding(input_ids) + self.pos_embedding(position_ids)
        # ...
```

## Files Modified

1. **`memory_test/test_algo/memory_profile_with_flash_attention.py`**
   - Changed `self.pos_embedding` from `nn.Parameter` to `nn.Embedding`
   - Updated `forward()` method to create position IDs and call `self.pos_embedding(position_ids)`

2. **`opacus/grad_sample/linear.py`**
   - Removed debug print statement at line 89

## Verification

After the fix:
- ✅ All 13 Linear layers in `SimpleBigModelWithFlashAttention` now have `_norm_sample` attributes
- ✅ `compute_linear_norm_sample` is called for all Linear layers (3D tensor path)
- ✅ Ghost Clipping + Flash Attention now works correctly
- ✅ Model training works normally with proper gradient norm clipping

## Key Takeaway

**When creating custom models for use with Opacus DP:**

⚠️ **Avoid defining parameters directly at the top level of your model using `nn.Parameter`**

Instead:
- ✅ Use `nn.Embedding`, `nn.Linear`, or other PyTorch modules
- ✅ Or wrap parameters in a submodule
- ✅ This ensures Opacus can properly recurse through your model's hierarchy and register hooks on all trainable submodules

This is a subtle but important requirement for compatibility with Opacus's hook-based gradient sampling system.

