# Using Opacus PrivacyEngine with facebook/DiT-XL-2-256

This guide explains how to train the [facebook/DiT-XL-2-256](https://huggingface.co/facebook/DiT-XL-2-256) diffusion model with Differential Privacy (DP-SGD) using Opacus's PrivacyEngine.

## Introduction

The DiT-XL-2-256 (Diffusion Transformer) is a large-scale transformer model for image generation. This implementation supports private training using the **flash_bk** mode (Flash Clipping with Bookkeeping), which provides:

- **Memory Efficiency**: Flash Attention reduces memory overhead compared to standard Ghost Clipping
- **Faster Training**: Optimized gradient computation with bookkeeping
- **Strong Privacy Guarantees**: Per-sample gradient clipping with formal DP guarantees

The flash_bk mode combines the best of both worlds: the memory efficiency of Flash Clipping and the performance benefits of bookkeeping.

## Installation & Setup

### Environment Setup

First, set up your environment:

```bash
source setup_env.sh
```

### Required Dependencies

```bash
# Core dependencies (already in requirements.txt)
pip install torch torchvision
pip install opacus
pip install diffusers
pip install transformers
pip install accelerate

# Optional: Flash Attention for additional memory optimization
pip install flash-attn
```

## Quick Start

Here's a minimal example using the flash_bk mode:

```python
import torch
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from memory_test.diffusion_profile.dit_huggingface_wrapper import DiTHuggingFaceWrapper

# 1. Create the DiT model
device = "cuda"
model = DiTHuggingFaceWrapper(
    model_name="facebook/DiT-XL-2-256",
    img_size=256,
    patch_size=2,
    in_channels=4,  # Latent space (not RGB!)
    num_classes=1000,
    pretrained=True,
    use_flash_attention=True,
).to(device)

# 2. Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 3. Create DataLoader (your latent-space dataset)
dataloader = DataLoader(your_dataset, batch_size=4, shuffle=True)

# 4. Create custom criterion for per-sample loss
def dit_criterion(predicted, target):
    batch_size = predicted.shape[0]
    pred_flat = predicted.reshape(batch_size, -1)
    target_flat = target.reshape(batch_size, -1)
    loss_per_element = torch.nn.functional.mse_loss(pred_flat, target_flat, reduction='none')
    return loss_per_element.mean(dim=1)  # Returns (B,) shape

dit_criterion.reduction = "mean"

# 5. Make model private with flash_bk mode
privacy_engine = PrivacyEngine()
model, optimizer, criterion, dataloader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=dataloader,
    criterion=dit_criterion,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
    grad_sample_mode="flash_bk",  # Flash Clipping + Bookkeeping
    poisson_sampling=True,
)

# 6. Training loop
for latent_images, timesteps, labels, target_noise in dataloader:
    latent_images = latent_images.to(device)
    timesteps = timesteps.to(device)
    labels = labels.to(device)
    target_noise = target_noise.to(device)
    
    # Forward pass
    predicted_noise = model(latent_images, timesteps, labels, target_noise=None)
    
    # Compute loss
    loss = criterion(predicted_noise, target_noise)
    
    # Backward and optimize
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## Critical Requirements

When using PrivacyEngine with DiT-XL-2-256, you **must** address these four requirements:

### 1. Freeze Tied Parameters

**Why**: Ghost Clipping and Flash Clipping do not support parameter tying (shared parameters). DiT models often have tied embedding layers.

**What happens if you don't**: You'll encounter the error:
```
NotImplementedError: Parameter tying is not supported with Ghost Clipping
```

**Solution**: The `DiTHuggingFaceWrapper` automatically detects and freezes tied parameters using the `_freeze_tied_parameters` method. This method:

1. Uses `data_ptr()` to reliably detect parameters sharing the same underlying tensor
2. Freezes all tied parameters by setting `requires_grad=False`
3. Also applies heuristics to freeze common embedding layers (timestep_embedder, class_embedder)

**Implementation reference** (from `dit_huggingface_wrapper.py`):

```python
def _freeze_tied_parameters(self, model):
    """Detect and freeze tied (shared) parameters to avoid Ghost Clipping issues."""
    data_ptr_to_names = {}
    param_list = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            data_ptr = param.data_ptr()
            if data_ptr not in data_ptr_to_names:
                data_ptr_to_names[data_ptr] = []
            data_ptr_to_names[data_ptr].append(name)
            param_list.append((name, param))
    
    # Find tied parameters (same underlying data)
    tied_params = {ptr: names for ptr, names in data_ptr_to_names.items() if len(names) > 1}
    
    if tied_params:
        for data_ptr, names in tied_params.items():
            for n, p in param_list:
                if p.data_ptr() == data_ptr and p.requires_grad:
                    p.requires_grad = False
```

**Note**: When using `DiTHuggingFaceWrapper` with `pretrained=True`, this is handled automatically.

### 2. Use DP-Compatible Layers

**Why**: Some PyTorch layers (e.g., `nn.MultiheadAttention`, BatchNorm) are incompatible with Opacus's per-sample gradient computation.

**What happens if you don't**: You'll encounter validation errors or incorrect gradient computation.

**Solution**: The `DiTHuggingFaceWrapper` uses Opacus's `ModuleValidator` to automatically fix incompatible layers:

```python
from opacus.validators import ModuleValidator

# Check for incompatible layers
errors = ModuleValidator.validate(model, strict=False)

# Fix incompatible layers (replaces with DP-compatible versions)
if len(errors) > 0 or use_flash_attention:
    fixed_model = ModuleValidator.fix(model, use_flash_attention=True)
    
# Verify the fix
assert ModuleValidator.is_valid(fixed_model)
```

**What it does**:
- Replaces `nn.MultiheadAttention` with `DPMultiheadAttention`
- Replaces BatchNorm/InstanceNorm with GroupNorm
- Applies Flash Attention optimizations when enabled

**Note**: When using `DiTHuggingFaceWrapper`, this is handled automatically during model initialization.

### 3. 4-Channel Latent Input Requirement

**Why**: The pretrained DiT-XL-2-256 model is trained on **latent space** representations (4 channels), not RGB images (3 channels).

**What happens if you don't**: You'll encounter the error:
```
RuntimeError: expected input[1, 3, 256, 256] to have 4 channels, but got 3 channels instead
```

**Solution**: You must preprocess RGB images into latent representations using a VAE (Variational Autoencoder) before feeding them to the DiT model.

#### Detailed VAE Preprocessing

Here's how to convert RGB images to 4-channel latents:

```python
import torch
from diffusers import AutoencoderKL

# 1. Load the VAE encoder
# Use the same VAE that DiT was trained with (typically Stable Diffusion VAE)
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",  # or "stabilityai/sd-vae-ft-ema"
    torch_dtype=torch.float32
).to(device)
vae.eval()  # Set to eval mode

# 2. Preprocessing function
def rgb_to_latent(rgb_images):
    """
    Convert RGB images to latent representations.
    
    Args:
        rgb_images: (B, 3, H, W) - RGB images in range [0, 1] or [-1, 1]
    
    Returns:
        latents: (B, 4, H//8, W//8) - Latent representations
    """
    # Normalize to [-1, 1] if needed
    if rgb_images.min() >= 0:
        rgb_images = rgb_images * 2.0 - 1.0
    
    # Encode to latent space
    with torch.no_grad():
        latent_dist = vae.encode(rgb_images).latent_dist
        latents = latent_dist.sample()  # or use .mean for deterministic
        
        # Apply scaling factor (standard for Stable Diffusion VAE)
        latents = latents * 0.18215
    
    return latents

# 3. Usage in training
# RGB images from your dataset
rgb_images = torch.randn(4, 3, 256, 256).to(device)  # (B, 3, 256, 256)

# Convert to latents
latent_images = rgb_to_latent(rgb_images)  # (B, 4, 32, 32)

# Now you can use latent_images with DiT
timesteps = torch.randint(0, 1000, (4,)).to(device)
labels = torch.randint(0, 1000, (4,)).to(device)

predicted_noise = model(latent_images, timesteps, labels)
```

#### Alternative: Use Pre-computed Latents

For efficiency, you can precompute latents before training:

```python
# Preprocess your entire dataset once
def preprocess_dataset(rgb_dataset, vae, device):
    latent_dataset = []
    
    for rgb_batch in rgb_dataset:
        latents = rgb_to_latent(rgb_batch.to(device))
        latent_dataset.append(latents.cpu())
    
    return latent_dataset

# Then use latent_dataset in your training loop
```

**Important Notes**:
- Input size: RGB (B, 3, 256, 256) → Latents (B, 4, 32, 32)
- The VAE downsamples by 8x: 256 → 32
- The scaling factor 0.18215 is standard for Stable Diffusion VAE
- You can also decode latents back to RGB using `vae.decode(latents / 0.18215)`

### 4. Custom Criterion for Per-Sample Loss

**Why**: Flash Clipping (flash_bk mode) requires per-sample loss computation. The criterion must return loss with shape `(B,)` instead of a scalar.

**What happens if you don't**: You'll encounter shape mismatch errors or incorrect privacy accounting.

**Solution**: Create a custom criterion that flattens spatial dimensions and computes per-sample MSE:

```python
import torch.nn as nn

def create_dit_criterion():
    """
    Create a criterion for DiT that computes per-sample MSE loss.
    Needed for PrivacyEngine with ghost/flash modes.
    """
    def dit_criterion(predicted, target):
        """
        Custom criterion for DiT that flattens outputs before computing loss.
        
        Args:
            predicted: (B, C, H, W) - predicted noise
            target: (B, C, H, W) - target noise
        
        Returns:
            loss_per_sample: (B,) - per-sample MSE loss
        """
        batch_size = predicted.shape[0]
        
        # Flatten spatial dimensions: (B, C, H, W) -> (B, C*H*W)
        pred_flat = predicted.reshape(batch_size, -1)
        target_flat = target.reshape(batch_size, -1)
        
        # Compute per-sample MSE and reduce over features
        loss_per_element = nn.functional.mse_loss(pred_flat, target_flat, reduction='none')
        return loss_per_element.mean(dim=1)  # (B,)
    
    # IMPORTANT: Set reduction attribute (required by PrivacyEngine)
    dit_criterion.reduction = "mean"
    
    return dit_criterion

# Usage
criterion = create_dit_criterion()

# In training loop
predicted_noise = model(latent_images, timesteps, labels, target_noise=None)
loss = criterion(predicted_noise, target_noise)  # Returns (B,) shape
loss.backward()  # PrivacyEngine handles per-sample gradients
```

**Key points**:
- Must flatten spatial dimensions before computing loss
- Must return shape `(B,)` not scalar
- Must set `criterion.reduction = "mean"` attribute
- Use `reduction='none'` in the MSE computation, then aggregate per-sample

## Complete Working Example

Here's a complete script that combines all four requirements:

```python
#!/usr/bin/env python3
"""
Complete example: Training DiT-XL-2-256 with DP-SGD (flash_bk mode)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from diffusers import AutoencoderKL
from opacus import PrivacyEngine
from memory_test.diffusion_profile.dit_huggingface_wrapper import DiTHuggingFaceWrapper

# ============================================================================
# 1. Setup: VAE for RGB -> Latent conversion
# ============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load VAE for preprocessing
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    torch_dtype=torch.float32
).to(device)
vae.eval()

def rgb_to_latent(rgb_images):
    """Convert RGB (B,3,256,256) to Latents (B,4,32,32)"""
    if rgb_images.min() >= 0:
        rgb_images = rgb_images * 2.0 - 1.0
    
    with torch.no_grad():
        latents = vae.encode(rgb_images).latent_dist.sample()
        latents = latents * 0.18215
    return latents

# ============================================================================
# 2. Dataset: Custom dataset that returns latents
# ============================================================================
class LatentDiffusionDataset(Dataset):
    """Dataset that provides latent representations and diffusion targets"""
    
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # In practice, load your pre-computed latents here
        # For this example, we'll generate synthetic latents
        latent = torch.randn(4, 32, 32)  # 4 channels, 32x32
        timestep = torch.randint(0, 1000, (1,)).item()
        label = torch.randint(0, 1000, (1,)).item()
        target_noise = torch.randn_like(latent)
        
        return latent, timestep, label, target_noise

# ============================================================================
# 3. Model: DiT with automatic tied parameter freezing and DP-compatible layers
# ============================================================================
model = DiTHuggingFaceWrapper(
    model_name="facebook/DiT-XL-2-256",
    img_size=256,
    patch_size=2,
    in_channels=4,  # CRITICAL: Must be 4 for latent space
    num_classes=1000,
    pretrained=True,  # Load pretrained weights
    use_flash_attention=True,  # Enable Flash Attention optimization
).to(device)

print(f"Model loaded. Tied parameters are automatically frozen.")
print(f"Model has been validated for Opacus compatibility.")

# ============================================================================
# 4. Custom Criterion: Per-sample loss computation
# ============================================================================
def create_dit_criterion():
    """Create criterion that returns per-sample loss (B,) shape"""
    def dit_criterion(predicted, target):
        batch_size = predicted.shape[0]
        pred_flat = predicted.reshape(batch_size, -1)
        target_flat = target.reshape(batch_size, -1)
        loss_per_element = nn.functional.mse_loss(pred_flat, target_flat, reduction='none')
        return loss_per_element.mean(dim=1)
    
    dit_criterion.reduction = "mean"  # REQUIRED attribute
    return dit_criterion

criterion = create_dit_criterion()

# ============================================================================
# 5. PrivacyEngine: Make model private with flash_bk mode
# ============================================================================
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
dataset = LatentDiffusionDataset(num_samples=10000)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

privacy_engine = PrivacyEngine()

print("\nMaking model private with flash_bk mode...")
model, optimizer, criterion, dataloader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=dataloader,
    criterion=criterion,
    noise_multiplier=1.0,      # Adjust for privacy/utility tradeoff
    max_grad_norm=1.0,         # Gradient clipping bound
    grad_sample_mode="flash_bk",  # Flash Clipping + Bookkeeping
    poisson_sampling=True,     # Use Poisson sampling for stronger privacy
)

print("Model is now private!")

# ============================================================================
# 6. Training Loop
# ============================================================================
num_epochs = 5

for epoch in range(num_epochs):
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (latents, timesteps, labels, target_noise) in enumerate(dataloader):
        # Move to device
        latents = latents.to(device)
        timesteps = timesteps.to(device)
        labels = labels.to(device)
        target_noise = target_noise.to(device)
        
        # Forward pass (returns tensor when target_noise=None)
        predicted_noise = model(latents, timesteps, labels, target_noise=None)
        
        # Handle 8-channel output (model outputs mean+var, we only need first 4)
        if predicted_noise.shape[1] > 4:
            predicted_noise = predicted_noise[:, :4, :, :]
        
        # Compute per-sample loss
        loss = criterion(predicted_noise, target_noise)
        
        # Backward (PrivacyEngine automatically handles per-sample gradients)
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.mean().item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.mean().item():.4f}")
    
    avg_loss = total_loss / num_batches
    
    # Get privacy spent
    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    
    print(f"Epoch {epoch} completed: Avg Loss = {avg_loss:.4f}, ε = {epsilon:.2f}")

print("\nTraining complete!")
print(f"Final privacy guarantee: (ε={epsilon:.2f}, δ=1e-5)")
```

## Other DP-SGD Modes

Besides `flash_bk`, Opacus supports several other grad_sample_mode options:

- **`ghost`**: Ghost Clipping - memory-efficient but slower than flash modes
- **`flash_clip`**: Flash Clipping - fast and memory-efficient, no bookkeeping
- **`bookkeeping`**: Traditional bookkeeping mode (compatible with more models)
- **`ghost_fsdp_bk`**: Ghost Clipping with FSDP for multi-GPU training

To compare different modes, use the provided shell script:

```bash
cd memory_test/diffusion_profile

# Run all experiments
bash run_all_experiments.sh

# Run specific experiments
bash run_all_experiments.sh --experiments vanilla,ghost,flash_clip,flash_clip_bk

# Run with different batch size
bash run_all_experiments.sh --batch-size 4 --num-iter 10
```

See `run_all_experiments.sh` for all available options.

## Troubleshooting

### Error: "Parameter tying is not supported with Ghost Clipping"

**Cause**: The model has tied (shared) parameters that haven't been frozen.

**Solution**: 
- If using `DiTHuggingFaceWrapper`, ensure `pretrained=True` is set (handles this automatically)
- If using raw diffusers model, manually freeze tied parameters:
```python
model = DiTHuggingFaceWrapper(pretrained=True, ...)  # Handles automatically
```

### Error: "expected input to have 4 channels, but got 3 channels"

**Cause**: Feeding RGB images directly instead of latent representations.

**Solution**: Use VAE encoder to preprocess images (see Section 4.3 above):
```python
latents = rgb_to_latent(rgb_images)  # Convert before feeding to model
```

### Error: Loss shape mismatch or "expected (B,) but got ()"

**Cause**: Criterion returns scalar loss instead of per-sample loss.

**Solution**: Use the custom criterion that flattens and returns (B,) shape:
```python
criterion = create_dit_criterion()  # Returns (B,) shape
criterion.reduction = "mean"  # Must set this attribute
```

### Warning: "size mismatch (8) must match size (4)"

**Cause**: DiT model outputs 8 channels (mean + variance) but target has 4 channels.

**Solution**: This is handled automatically in `DiTHuggingFaceWrapper.forward()`:
```python
if predicted_noise.shape[1] > self.in_channels:
    predicted_noise = predicted_noise[:, :self.in_channels, :, :]
```

If using raw model, add this slicing manually.

### Error: "ModuleValidator found incompatible layers"

**Cause**: Model contains layers incompatible with Opacus.

**Solution**: The wrapper fixes this automatically. If you see this, ensure:
```python
model = DiTHuggingFaceWrapper(pretrained=True, use_flash_attention=True, ...)
# Automatically applies ModuleValidator.fix()
```

## References

- **Implementation**: See `single_experiment.py` for complete experiment code
- **Model Wrapper**: See `dit_huggingface_wrapper.py` for DiT model details
- **Batch Scripts**: See `run_all_experiments.sh` for running multiple experiments
- **Opacus Documentation**: https://opacus.ai/
- **DiT Paper**: https://arxiv.org/abs/2212.09748
- **Diffusers Library**: https://huggingface.co/docs/diffusers

## File Structure

```
memory_test/diffusion_profile/
├── README.md                      # This file
├── dit_huggingface_wrapper.py     # DiT model wrapper with DP support
├── single_experiment.py           # Run individual experiments
├── run_all_experiments.sh         # Batch experiment runner
├── visualize_memory_breakdown.py  # Visualize memory profiling results
└── inspect_dit_model.py           # Inspect model structure
```

## Quick Command Reference

```bash
# Setup environment
source setup_env.sh

# Run single experiment with flash_bk
cd memory_test/diffusion_profile
python single_experiment.py \
    --experiment flash_clip_bk \
    --output results.json \
    --batch-size 4 \
    --num-iter 10 \
    --use-flash-attention

# Run all experiments
bash run_all_experiments.sh --batch-size 4

# Run without Flash Attention
bash run_all_experiments.sh --no-flash-attention

# Inspect model structure
python inspect_dit_model.py
```

## Citation

If you use this implementation, please cite:

```bibtex
@article{pmlr-v139-peebles21a,
  title={Scalable Diffusion Models with Transformers},
  author={Peebles, William and Xie, Saining},
  journal={arXiv preprint arXiv:2212.09748},
  year={2022}
}

@inproceedings{opacus,
  title={Opacus: User-Friendly Differential Privacy Library in PyTorch},
  author={Ashkan Yousefpour and others},
  booktitle={arXiv preprint arXiv:2109.12298},
  year={2021}
}
```

