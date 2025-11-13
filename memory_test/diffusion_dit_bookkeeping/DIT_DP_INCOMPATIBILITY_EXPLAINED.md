# DiT + Opacus DP-SGD: Incompatibility Explanation

## The Fundamental Problem

### 1. How Opacus Computes Per-Sample Gradients

Opacus uses PyTorch's `functorch` library to compute per-sample gradients efficiently:

```python
# Simplified version of Opacus's per-sample gradient computation
def compute_per_sample_grads(model, batch_input):
    def compute_single_sample_grad(single_input):
        # Run model on ONE sample
        output = model(single_input)
        loss = criterion(output, target)
        # Compute gradients for this ONE sample
        grads = torch.autograd.grad(loss, model.parameters())
        return grads
    
    # V map: vectorize over batch dimension
    # Transforms model(batch) → [model(sample1), model(sample2), ...]
    per_sample_grads = vmap(compute_single_sample_grad)(batch_input)
    return per_sample_grads
```

**Key Constraint**: `vmap` only vectorizes the **first argument** of `forward()`.

### 2. DiT's Forward Signature

```python
class DiTModel:
    def forward(self, x, t, y):
        # x: images (B, C, H, W)
        # t: timesteps (B,) - CONDITIONAL
        # y: labels (B,) - CONDITIONAL
        ...
```

**The Problem**: When Opacus calls `vmap(model)(batch_x)`, it only passes `x`. The arguments `t` and `y` are not passed, causing:

```
TypeError: forward() missing 2 required positional arguments: 't' and 'y'
```

###3. Why Embedding Conditionals Doesn't Fully Work

Even if we try to embed `t` and `y` into the input tensor:

```python
x_combined = torch.cat([images, timestep_map, label_map], dim=1)
# Inside forward:
t = x_combined[:, 3, 0, 0].long()  # Extract timestep
y = x_combined[:, 4, 0, 0].long()  # Extract label
```

**Problem**: When `vmap` splits the batch:
- `x_combined` becomes a **single sample** tensor (no batch dimension)
- Indexing `[:, 3, 0, 0]` no longer makes sense
- The embedding lookup `self.timestep_embedder(t)` expects a scalar but gets wrong shape

This is why we still get errors in the `functorch` gradient computation.

## Why This Is Fundamentally Incompatible

The core issue: **Conditional generative models require per-sample conditional information that must be processed differently for each sample, but Opacus's `vmap` assumes the model only needs one input tensor.**

DiT specifically requires:
1. **Per-sample timesteps**: Different diffusion timestep for each image
2. **Per-sample labels**: Different class labels for each image
3. **Conditional embeddings**: These must be computed per-sample before being used in attention blocks

When `vmap` batches these operations, the conditional flow breaks because:
- Embedding lookups expect specific tensor shapes
- The conditioning vector is computed per-sample but must broadcast correctly
- AdaLN modulation requires per-sample scale/shift parameters

## Actual Solutions

### Solution 1: Remove Conditional Inputs (Simplest)

Train an **unconditional** DiT model:

```python
class UnconditionalDiT(nn.Module):
    def __init__(self, ...):
        # No timestep_embedder
        # No label_embedder
        ...
    
    def forward(self, x):
        # x: images only
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        # No conditioning vector needed
        for block in self.blocks:
            x = block(x)  # Standard transformer block (no adaLN)
        
        return self.final_layer(x)
```

**Pros**: Works perfectly with Opacus
**Cons**: Can't do conditional generation (no timestep/class control)

### Solution 2: Use Opacus's "Expand" Mode (If Available)

Some versions of Opacus support passing additional arguments to `forward()` via hooks, but this is **not standard** and requires:
- Modifying Opacus core
- Custom gradient computation
- Deep understanding of functorch

### Solution 3: Use Microbatching Instead of Per-Sample Gradients

Instead of computing per-sample gradients with `functorch`, use **physical microbatching**:

```python
# Pseudocode
for microbatch in split_into_microbatches(batch, microbatch_size=1):
    img, t, y = microbatch
    pred = model(img, t, y)
    loss = criterion(pred, target)
    loss.backward()  # Accumulate gradients
    # Clip per-microbatch gradients
    clip_gradients(model, max_norm)
optimizer.step()
```

**Pros**: Works with any model architecture
**Cons**:
- Much slower (B separate backward passes)
- Higher memory overhead
- Doesn't leverage Opacus's optimizations

### Solution 4: Implement Custom Per-Sample Gradient Computation

Write custom backward hooks that understand DiT's structure:

```python
class CustomDiTGradSampler:
    def compute_per_sample_grads(self, model, images, timesteps, labels):
        # Manually implement per-sample gradient computation
        # for each layer type in DiT (Linear, LayerNorm, etc.)
        ...
```

**Pros**: Most flexible
**Cons**:
- Extremely complex (100s of lines)
- Hard to maintain
- Must handle all layer types
- Performance may not match functorch

### Solution 5: Use a Different DP-SGD Library

Some alternatives that might handle conditional models better:
- **Private Transformers** (Hugging Face): Has better conditional support
- **JAX + objax_privacy**: Different approach to per-sample gradients
- **TensorFlow Privacy**: Different architecture

## Recommended Approach for Your Use Case

Given you want to profile **memory usage** of DiT with DP-SGD:

### Option A: Profile Vanilla DiT Only
Run memory profiling on the **vanilla (non-DP) DiT model** to understand baseline memory usage. This is valuable even without DP-SGD.

```bash
python single_experiment.py --experiment vanilla --output vanilla_result.json
```

### Option B: Simplify DiT for DP Compatibility
Create a **simplified unconditional DiT** specifically for DP-SGD profiling:

1. Remove timestep and label inputs
2. Use standard transformer blocks (no adaLN)
3. Profile this simplified version with DP-SGD
4. Results approximate the memory overhead of DP-SGD on transformer architectures

### Option C: Use LLM Results as Proxy
The LLM experiments in `fastdp_bookkeeping` use similar transformer architecture. Memory patterns should be similar:
- Both use multi-head attention
- Both use feed-forward networks
- DP-SGD overhead should be comparable

## Technical Deep Dive: Why `vmap` Fails

```python
# What we want:
batch_images = [img1, img2]  # B=2
batch_t = [t1, t2]           # Different timesteps
batch_y = [y1, y2]           # Different labels

# What vmap does:
for img, t, y in zip(batch_images, batch_t, batch_y):
    # Process ONE sample at a time
    t_emb = timestep_embedder(t)  # t is scalar
    y_emb = label_embedder(y)      # y is scalar
    c = t_emb + y_emb
    ...

# What actually happens in vmap:
# vmap splits ONLY the first argument (images)
# t and y are missing → ERROR

# Even if we concatenate:
x_combined = cat([images, expand(t), expand(y)])
# When vmap processes single sample:
single_x = x_combined[0]  # Shape changes!
# Extracting t from single_x doesn't work as expected
```

## Conclusion

**DiT's conditional architecture (timestep + label) is fundamentally incompatible with Opacus's functorch-based per-sample gradient computation** because:

1. `vmap` only handles single-input `forward()` methods
2. Conditional inputs require per-sample processing that breaks under `vmap`
3. Embedding lookups and conditional flows don't vectorize correctly

**Practical recommendation**: Focus on profiling vanilla DiT, or use LLM results as a proxy for DP-SGD memory overhead on transformer architectures.

