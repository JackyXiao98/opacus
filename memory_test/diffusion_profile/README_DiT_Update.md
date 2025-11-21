# DiT Model Update: facebook/DiT-XL-2-256

## Overview

The code has been updated to use `facebook/DiT-XL-2-256` from the `diffusers` library instead of `microsoft/dit-large`. This provides the standard DiT (Diffusion Transformer) architecture.

## Key Changes

### Architecture Parameters (DiT-XL-2-256)
- **Image size**: 256×256 (down from 1024×1024)
- **Patch size**: 2×2 (down from 8×8)
- **Input channels**: 4 (latent space) or 3 (RGB)
- **Hidden dimension**: 1152
- **Number of layers**: 28
- **Number of attention heads**: 16

### Important Notes

#### Pretrained vs Custom Implementation

**Option 1: Custom DP-Compatible Implementation (Recommended for experiments)**
```python
model = DiTHuggingFaceWrapper(
    model_name="facebook/DiT-XL-2-256",
    img_size=256,
    patch_size=2,
    in_channels=3,  # Can be 3 (RGB) or 4 (latent)
    num_classes=1000,
    pretrained=False,  # Uses custom DP-compatible implementation
)
```
- ✅ Works with any number of input channels (3 for RGB, 4 for latent)
- ✅ Fully compatible with Opacus DP-SGD
- ✅ No additional dependencies needed
- ⚠️ Not using pretrained weights

**Option 2: Pretrained Diffusers Model (Advanced)**
```python
model = DiTHuggingFaceWrapper(
    model_name="facebook/DiT-XL-2-256",
    img_size=256,
    patch_size=2,
    in_channels=4,  # MUST be 4 for pretrained model
    num_classes=1000,
    pretrained=True,  # Loads pretrained weights from diffusers
)
```
- ⚠️ **REQUIRES 4-channel latent space inputs** (not RGB images)
- ⚠️ Expects inputs from VAE encoder
- ✅ Uses pretrained weights
- ⚠️ May have Opacus compatibility issues

## Installation

```bash
source .venv/bin/activate
uv pip install diffusers
```

## Usage Examples

### Running Experiments

```bash
# Vanilla experiment with 4-channel latent inputs
python memory_test/diffusion_profile/single_experiment.py \
    --experiment vanilla \
    --output results_vanilla.json \
    --image-size 256 \
    --patch-size 2 \
    --in-channels 4 \
    --batch-size 1 \
    --num-iter 3

# Experiment with 3-channel RGB inputs
python memory_test/diffusion_profile/single_experiment.py \
    --experiment ghost \
    --output results_ghost.json \
    --image-size 256 \
    --patch-size 2 \
    --in-channels 3 \
    --batch-size 1 \
    --num-iter 3
```

### Inspecting the Model

```bash
python memory_test/diffusion_profile/inspect_dit_model.py
```

## Architecture Comparison

| Parameter | microsoft/dit-large | facebook/DiT-XL-2-256 |
|-----------|-------------------|---------------------|
| Image Size | 1024×1024 | 256×256 |
| Patch Size | 8×8 | 2×2 |
| Input Channels | 3 (RGB) | 4 (latent) |
| Hidden Dim | 1024 | 1152 |
| Layers | 24 | 28 |
| Heads | 16 | 16 |

## Troubleshooting

### Error: "expected input to have 4 channels, but got 3 channels"
**Solution**: Set `pretrained=False` in the model initialization, or use 4-channel inputs.

### Error: "diffusers library not found"
**Solution**: Install diffusers with `uv pip install diffusers`

### Memory issues
**Solution**: Reduce batch size or image size parameters.

## Reference

For more information about DiT models, see:
- [DiT Paper](https://arxiv.org/abs/2212.09748)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [facebook/DiT-XL-2-256 Model Card](https://huggingface.co/facebook/DiT-XL-2-256)

