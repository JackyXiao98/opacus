# DiT-L + Flash Attention Memory Profiling

This directory contains memory profiling experiments for **Diffusion Transformer (DiT-L)** with **Flash Attention**, adapted from the FastDP bookkeeping implementation.

## Overview

This experiment suite evaluates memory usage and performance of DiT-L (Large variant) under different DP-SGD training configurations. The goal is to compare memory footprint and training time across:

1. **Vanilla (Non-DP)**: Standard training without differential privacy
2. **Ghost Clipping**: DP-SGD with 2-pass gradient computation (no per-sample gradients stored)
3. **Flash Clipping**: DP-SGD with Triton-accelerated gradient clipping
4. **Bookkeeping**: DP-SGD with single-pass bookkeeping optimization
5. **Flash Clipping + Bookkeeping**: Combined Triton acceleration with bookkeeping

## Model Architecture: DiT-L

**Diffusion Transformer - Large Configuration**

- **Hidden Dimension**: 1024
- **Number of Layers**: 24
- **Number of Attention Heads**: 16
- **Image Size**: 256 × 256 pixels
- **Patch Size**: 8 × 8 pixels
- **Number of Tokens**: 1024 (32×32 patches)
- **Input Channels**: 3 (RGB)
- **Number of Classes**: 1000 (ImageNet-style)

### Architecture Components

1. **PatchEmbed**: Conv2d layer that converts images into patch embeddings
2. **Positional Embedding**: Learnable position embeddings for spatial information
3. **TimestepEmbedder**: Sinusoidal + MLP embedding for diffusion timestep
4. **LabelEmbedder**: Class label embedding with dropout for classifier-free guidance
5. **DiT Blocks** (24 layers):
   - Flash Attention (via `DPMultiheadAttentionWithFlashAttention`)
   - Adaptive Layer Normalization (adaLN-Zero) with timestep/label conditioning
   - MLP with GELU activation
6. **Final Layer**: Projects to noise prediction in patch space

### Training Task

The model is trained for **diffusion-based image generation**:
- **Input**: Noisy images (B, 3, 256, 256), timesteps (B,), class labels (B,)
- **Output**: Predicted noise (B, 3, 256, 256)
- **Loss**: MSE between predicted and target noise

## Files

### Core Implementation

- **`diffusion_dit_model.py`**: DiT-L model definition with Flash Attention
  - `DiTModelWithFlashAttention`: Main model class
  - `DiTBlockWithFlashAttention`: Transformer block with adaLN conditioning
  - `PatchEmbed`, `TimestepEmbedder`, `LabelEmbedder`: Supporting modules

- **`single_experiment.py`**: Script to run individual experiments
  - `run_vanilla_experiment()`: Non-DP baseline
  - `run_dpsgd_experiment()`: DP-SGD variants (ghost/flash/bookkeeping)

- **`run_all_experiments.sh`**: Shell script to run all 5 experiments sequentially
  - Runs each experiment in isolated Python process
  - Prevents memory contamination between experiments
  - Generates visualizations automatically

- **`visualize_memory_breakdown.py`**: Visualization and analysis script
  - Memory breakdown by component
  - Memory timeline across training steps
  - Performance trade-off plots
  - Summary statistics

## Usage

### Running Experiments

**Run all experiments:**

```bash
cd memory_test/diffusion_dit_bookkeeping
bash run_all_experiments.sh
```

This will:
1. Create a timestamped output directory
2. Run all 5 experiments sequentially
3. Generate JSON results for each experiment
4. Create visualizations comparing results
5. Print summary statistics

**Run a single experiment:**

```bash
python single_experiment.py \
    --experiment vanilla \
    --output results/vanilla.json \
    --image-size 256 \
    --patch-size 8 \
    --hidden-dim 1024 \
    --num-layers 24 \
    --num-heads 16 \
    --batch-size 2 \
    --num-iter 3 \
    --warmup-iter 0
```

Available experiments: `vanilla`, `ghost`, `flash_clip`, `bookkeeping`, `flash_clip_bookkeeping`

### Configuration

Default configuration in `run_all_experiments.sh`:

```bash
IMAGE_SIZE=256        # Image resolution
PATCH_SIZE=8          # Patch size for embedding
IN_CHANNELS=3         # RGB images
NUM_CLASSES=1000      # Number of class labels
HIDDEN_DIM=1024       # DiT-L hidden dimension
NUM_LAYERS=24         # DiT-L number of layers
NUM_HEADS=16          # DiT-L attention heads
BATCH_SIZE=2          # Batch size
NUM_ITER=3            # Number of profiling iterations
WARMUP_ITER=0         # Warmup iterations (skipped)
```

**Number of Tokens**: (IMAGE_SIZE / PATCH_SIZE)² = (256/8)² = 1024 ✓

### Output Structure

```
memory_profiling_results/
└── run_YYYYMMDD_HHMMSS/
    ├── vanilla_result.json
    ├── ghost_result.json
    ├── flash_clip_result.json
    ├── flash_clip_bookkeeping_result.json
    ├── bookkeeping_result.json
    └── visualizations/
        ├── memory_breakdown_comparison.png
        ├── memory_timeline.png
        ├── performance_tradeoff.png
        └── summary.txt
```

Each JSON result contains:
- `experiment`: Experiment name
- `config`: Model/training configuration
- `peak_memory_mb`: Peak CUDA memory usage
- `avg_time_ms`: Average iteration time
- `breakdown`: Detailed component-wise memory breakdown
- `snapshots`: Memory usage at each training stage

## Results Interpretation

### Memory Components

- **Model Parameters**: Weights of the DiT model
- **Optimizer States**: Adam optimizer state (momentum, variance)
- **Gradients**: Parameter gradients
- **Activation Hooks**: Saved activations for DP-SGD (when applicable)
- **Norm Samples**: Per-sample gradient norms for clipping
- **Temp Matrices**: Temporary matrices (ggT, aaT) during gradient computation

### Expected Trends

1. **Vanilla < Ghost/Flash < Bookkeeping**
   - Vanilla has lowest memory (no DP overhead)
   - Ghost/Flash use 2-pass or optimized clipping
   - Bookkeeping caches activations for single pass

2. **Ghost ≈ Flash (memory), Flash < Ghost (time)**
   - Flash Clipping uses Triton kernels for speedup
   - Similar memory footprint to Ghost Clipping

3. **Bookkeeping + Flash Clip = Best Time/Memory Trade-off**
   - Single-pass with Triton acceleration
   - Balanced memory and speed

## Comparison with LLM Experiments

This DiT experiment parallels `memory_test/fastdp_bookkeeping` but with key differences:

| Aspect | LLM (fastdp_bookkeeping) | DiT (diffusion_dit_bookkeeping) |
|--------|-------------------------|--------------------------------|
| **Architecture** | Transformer decoder | Diffusion Transformer |
| **Input** | Token IDs (B, L) | Images (B, 3, H, W) |
| **Task** | Language modeling | Diffusion denoising |
| **Loss** | CrossEntropyLoss | MSELoss |
| **Conditioning** | None | Timestep + class label |
| **Sequence Length** | 8192 tokens | 1024 tokens (patches) |
| **Attention** | Causal | Bidirectional |

Both use:
- Flash Attention for memory efficiency
- Same DP-SGD methods (Ghost/Flash/Bookkeeping)
- Same profiling infrastructure

## Requirements

- PyTorch 2.0+ (for Flash Attention support)
- CUDA GPU (required for memory profiling)
- Opacus with FastDP bookkeeping support
- matplotlib, numpy (for visualization)

## Testing on CPU

The code will run on CPU but:
- Memory statistics will show 0 (no CUDA tracking)
- Primary purpose is syntax validation
- Triton kernels will fall back to PyTorch
- Performance metrics will not be representative

## Troubleshooting

### Out of Memory

If experiments fail with OOM:
- Reduce `BATCH_SIZE` in `run_all_experiments.sh`
- Reduce `NUM_LAYERS` (e.g., try DiT-S with 12 layers)
- Reduce `IMAGE_SIZE` (e.g., 128×128 → 256 tokens)

### Slow Execution

- Reduce `NUM_ITER` (default: 3)
- Skip Ghost/Bookkeeping experiments (slower due to 2-pass)
- Use smaller model variant

### Missing Dependencies

Ensure you have the custom modules:
```python
from memory_test.test_algo.memory_profile_with_flash_attention import DPMultiheadAttentionWithFlashAttention
from memory_test.test_algo.detailed_memory_profiler import EnhancedMemoryProfiler
```

## References

- **DiT Paper**: [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
- **Flash Attention**: [Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- **FastDP**: [Book-Keeping for Fast and Accurate DP-SGD](https://arxiv.org/abs/2103.01624)
- **Ghost Clipping**: [Efficient Gradient Clipping via Gradient Norms](https://arxiv.org/abs/2110.05679)

## Future Enhancements

- [ ] Test on larger models (DiT-XL/2)
- [ ] Vary sequence length (different image/patch sizes)
- [ ] Add gradient accumulation support
- [ ] Benchmark on multiple GPUs
- [ ] Compare with standard attention (non-Flash)

## Contact

For issues or questions, please refer to the main Opacus repository or the FastDP bookkeeping documentation.

