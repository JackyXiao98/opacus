Flash-Norm
Constant-Memory Gradient Primitives for Long-Context Private Training

Flash-Norm is a high-performance, I/O-aware gradient primitive designed to overcome the memory bottlenecks in Differentially Private Stochastic Gradient Descent (DP-SGD). It enables training of Large Language Models (LLMs) and other foundation models on extremely long sequences (e.g., 128k) by computing exact per-sample gradient norms with $$O(1)$$ intermediate memory complexity.


---

🔥 Key Features

- ⚡️ $$O(1)$$ Memory Overhead: Computes gradient norms without materializing the $$O(Bdp)$$ gradient matrix (like Opacus) or the $$O(T^2)$$ Gram matrix (like GhostClip).
- 🚀 TMA-Accelerated: Leverages NVIDIA Hopper (H100) Tensor Memory Accelerator (TMA) via `tl.make_block_ptr` for asynchronous data movement, hiding memory latency.
- 🧩 Split-T Parallelism: Maximizes GPU occupancy for long sequences by parallelizing T-dimension reduction across multiple thread blocks with atomic barrier synchronization.
- 🔧 Unified Workflow: Supports both 1-Pass (Bookkeeping mode, single-pass clipping with cached activations) and 2-Pass (GhostClipping mode).
- ✅ Exact & rigorous: Provides mathematically exact gradient norms (FP32 accumulation), unlike approximate methods like per-layer clipping.
---

📈 Performance

- Breaking the Memory Wall: Flash-Norm maintains a constant, negligible memory footprint regardless of sequence length $$T$$ or batch size $B$.
- Privacy for Free: By enabling larger batch sizes (e.g., $$B=32$$ vs $B=4$), Flash-Norm achieves training throughput comparable to standard non-private SGD.
  

---

🛠️ Installation

Requirements:
- NVIDIA GPU (Hopper H100 recommended for TMA support; Ampere A100 supported with fallback)
- PyTorch >= 2.8
- Triton >= 2.1
- CUDA >= 12.0
  
pip install flash-norm
```

Or build from source:

```bash
git clone https://github.com/your-repo/flash-norm.git
cd flash-norm
pip install -e .
```


---

🚀 Usage

Flash-Norm provides `FusedFlashLinear` as a drop-in replacement for `nn.Linear` that computes per-sample gradient norms directly in the backward pass.

### 1. Basic Usage with Privacy Engine

The easiest way to use Flash-Norm is through `FlashNormPrivacyEngine`, which automatically handles the integration:

```python
import torch
from flashnorm.privacy_engine import FlashNormPrivacyEngine

model = YourModel()  # Your PyTorch model
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

# Initialize Privacy Engine
privacy_engine = FlashNormPrivacyEngine()
model, optimizer, criterion, dataloader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=dataloader,
    noise_multiplier=0.5,
    max_grad_norm=1.0,
    grad_sample_mode="flash",  # Use Flash-Norm kernels
)

# Training Loop
for batch in dataloader:
    optimizer.zero_grad()
    outputs = model(**batch)
    loss = criterion(outputs, batch['labels'])
    
    # Flash-Norm handles norm computation internally
    loss.backward() 
    optimizer.step()
```

### 2. Direct Usage with FusedFlashLinear

For more control, you can directly use `FusedFlashLinear` modules:

```python
import torch
import torch.nn as nn
from flashnorm.grad_sample.fused_flash_linear import (
    FusedFlashLinear,
    replace_linear_with_fused,
    get_fused_linear_modules,
)

# Option 1: Replace Linear layers in an existing model
model = YourModel()
model = replace_linear_with_fused(model)

# Option 2: Use FusedFlashLinear directly
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = FusedFlashLinear(768, 3072, bias=True)
        self.linear2 = FusedFlashLinear(3072, 768, bias=True)
    
    def forward(self, x):
        return self.linear2(self.linear1(x))

model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Setup norm buffer for per-sample norm accumulation
batch_size = 32
norm_buf = torch.zeros(batch_size, device='cuda')

# Get all FusedFlashLinear modules and configure them
fused_modules = get_fused_linear_modules(model)
for module in fused_modules:
    module.set_norm_buffer(norm_buf)
    module.set_compute_norms(True)

# Training loop
for batch in dataloader:
    optimizer.zero_grad()
    norm_buf.zero_()  # Reset norm buffer
    
    outputs = model(batch['inputs'])
    loss = criterion(outputs, batch['labels'])
    
    loss.backward()
    
    # After backward, norm_buf contains per-sample squared norms
    per_sample_norms = torch.sqrt(norm_buf + 1e-8)
    print(f"Per-sample norms: {per_sample_norms}")
    
    optimizer.step()
```

### 3. Bookkeeping Mode (Single-Pass Clipping)

Bookkeeping mode caches activations and gradients during backward, then computes clipped gradients manually. This eliminates the need for a second backward pass:

```python
# Enable bookkeeping mode before forward
fused_modules = get_fused_linear_modules(model)
for module in fused_modules:
    module.set_norm_buffer(norm_buf)
    module.set_compute_norms(True)
    module.set_bookkeeping_mode(True)  # Enable bookkeeping

# Forward and backward (caches activations/gradients)
optimizer.zero_grad()
norm_buf.zero_()
outputs = model(batch['inputs'])
loss = criterion(outputs, batch['labels'])
loss.backward()  # Computes norms and caches x/grad_out

# Compute clipping coefficients from norms
per_sample_norms = torch.sqrt(norm_buf + 1e-8)
clipping_coef = (1.0 / per_sample_norms).clamp(max=1.0)

# Zero gradients (from non-clipped backward)
optimizer.zero_grad()

# Compute clipped gradients from cached values
for module in fused_modules:
    module.compute_clipped_gradient(
        clipping_coef, 
        grad_scale=1.0  # Use batch_size if loss_reduction="mean"
    )
    module.clear_bk_cache()  # Free cached memory

optimizer.step()
```

### 4. Input Dimension Support

`FusedFlashLinear` automatically handles different input dimensions:

```python
# 2D inputs: [B, Din]
x_2d = torch.randn(32, 768, device='cuda')
out_2d = fused_linear(x_2d)

# 3D inputs: [B, T, Din] (e.g., sequences)
x_3d = torch.randn(32, 512, 768, device='cuda')
out_3d = fused_linear(x_3d)

# 4D+ inputs: [B, d1, d2, ..., Din] (e.g., images)
x_4d = torch.randn(32, 3, 224, 224, device='cuda')
# Automatically reshaped to [B, T, Din] where T = 3*224*224
out_4d = fused_linear(x_4d)
```


---

🧠 How It Works

### Register-Centric Strip-Mining

Instead of writing intermediate results to HBM (which is slow and memory-heavy), Flash-Norm tiles the computation along the sequence length $$T$$ and accumulates partial results directly in GPU Registers.

For a Linear layer with input $$X \in \mathbb{R}^{B \times T \times D_{in}}$$ and output gradient $$G \in \mathbb{R}^{B \times T \times D_{out}}$$, the per-sample weight gradient is:

$$\mathbf{G}_b = \mathbf{G}_b^T \mathbf{X}_b \in \mathbb{R}^{D_{out} \times D_{in}}$$

The per-sample gradient norm squared is:

$$\|\mathbf{G}_b\|_F^2 = \sum_{i,j} (\mathbf{G}_b)_{i,j}^2$$

Flash-Norm computes both the aggregated gradient $$\sum_b \mathbf{G}_b$$ and the per-sample norms $$\|\mathbf{G}_b\|_F^2$$ in a single fused kernel pass, without materializing the $$O(B \cdot D_{out} \cdot D_{in})$$ per-sample gradient matrix.

### Fused Backward Computation

The Triton kernel fuses two operations:

1. **Weight Gradient**: $$\text{grad\_weight} = \sum_{b=1}^{B} \mathbf{G}_b^T \mathbf{X}_b$$
2. **Per-Sample Norms**: $$\text{norms}[b] = \|\mathbf{G}_b^T \mathbf{X}_b\|_F^2$$

This is achieved by:
- Tiling along $$T$$ dimension with block size `BLOCK_K`
- Accumulating partial results in registers
- Computing norm contributions tile-by-tile
- Using atomic operations to accumulate norms across tiles

### Hardware-Aware Optimizations

**TMA Block Pointers (H100)**
- Uses `tl.make_block_ptr` for hardware-accelerated memory access
- Asynchronous data movement via TMA (Tensor Memory Accelerator)
- Automatic boundary checking and masking

**Autotuning**
- Dynamically selects optimal block sizes (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`) based on input dimensions
- Configures `num_stages` for software pipelining (4-5 stages for H100)
- Optimizes `num_warps` for GPU occupancy

**TF32 Support**
- Explicitly enables TF32 for Tensor Core acceleration on Ampere+ GPUs
- Maintains FP32 accumulation for numerical accuracy

**Split-K Parallelism**
- For large sequences (T >= 512), automatically splits T-dimension across multiple thread blocks
- Uses atomic barriers for inter-block synchronization
- Leader block aggregates partial results from non-leader blocks
- Provides speedup for long sequences by increasing parallelism

### Bookkeeping Mode Algorithm

Bookkeeping mode enables single-pass clipping:

1. **Forward Pass**: Standard forward computation
2. **Backward Pass**: 
   - Compute per-sample gradient norms
   - Cache input activations $$X$$ and output gradients $$G$$
   - Skip gradient computation (return `None` for `grad_w`)
3. **Clipping**: Compute clipping coefficients from norms
4. **Manual Gradient Computation**: 
   $$\text{clipped\_grad\_w} = \sum_b c_b \cdot \mathbf{G}_b^T \mathbf{X}_b = (\mathbf{C} \odot \mathbf{G})^T \mathbf{X}$$
   
   where $$c_b$$ is the clipping coefficient for sample $$b$$, and $$\mathbf{C}$$ is a diagonal matrix of coefficients.

This eliminates the second backward pass required by standard ghost clipping.


---

🔧 Advanced Configuration

### Hardware Detection

Flash-Norm automatically detects GPU capabilities:

```python
from flashnorm.grad_sample.triton_fused_kernel import (
    is_hopper_gpu,
    has_dsmem_support,
)

# Check if running on H100/Hopper GPU
if is_hopper_gpu():
    print("Hopper GPU detected - TMA optimizations enabled")

# Check if Triton supports DSMEM/cluster APIs
if has_dsmem_support():
    print("DSMEM support available")
```

### CPU Fallback

When CUDA/Triton is not available, Flash-Norm falls back to efficient CPU computation:

- 2D inputs: Uses rank-1 outer product property: $$\|\mathbf{g}_i \mathbf{x}_i^T\|_F^2 = \|\mathbf{g}_i\|^2 \|\mathbf{x}_i\|^2$$
- 3D+ inputs: Materializes per-sample gradients for norm computation

### Split-K Configuration

Split-K is automatically enabled for large T. The kernel selects `split_k` based on sequence length:

- T >= 1024: `split_k = 4` (higher parallelism)
- T >= 512: `split_k = 2` (balanced)
- T < 512: Standard kernel (no split)

The Split-K kernel uses atomic barriers for synchronization:
- Non-leader blocks write partial results to global buffer and signal via atomic counter
- Leader block waits for all signals, aggregates partials, computes norm, resets barrier
- This provides portable inter-block communication without requiring DSMEM hardware


---

📚 API Reference

### FusedFlashLinear

**Methods:**
- `set_norm_buffer(norm_buf: torch.Tensor)`: Set buffer for accumulating per-sample squared norms
- `set_compute_norms(compute: bool)`: Enable/disable norm computation in backward
- `set_bookkeeping_mode(enable: bool)`: Enable bookkeeping mode for single-pass clipping
- `compute_clipped_gradient(clipping_coef: torch.Tensor, grad_scale: float)`: Compute clipped gradients from cached values (bookkeeping mode)
- `clear_bk_cache()`: Clear bookkeeping cache to free memory

**Utility Functions:**
- `replace_linear_with_fused(module: nn.Module)`: Recursively replace `nn.Linear` with `FusedFlashLinear`
- `get_fused_linear_modules(module: nn.Module)`: Get all `FusedFlashLinear` modules in a model

For detailed API documentation, see the implementation in `flashnorm/grad_sample/fused_flash_linear.py`.


---

📄 Citation

If you use Flash-Norm in your research, please cite our paper:

```bibtex
@article{flashnorm2025,
  title={Flash-Norm: Constant-Memory Gradient Primitives for Long-Context Private Training},
  author={Anonymous Authors},
  journal={Under Review},
  year={2025}
}
```


---

📄 License

This project is licensed under the Apache-2.0 License.
