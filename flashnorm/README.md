# FlashNorm

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

## Introduction

**FlashNorm** is an advanced differentially private deep learning library built on PyTorch. It enables training machine learning models with formal privacy guarantees while maintaining high performance and usability. This fork introduces significant optimizations to accelerate private training and reduce memory overhead through novel algorithmic innovations and hardware-aware kernel optimizations.

### Key Features

- **Flash Norm Clipping Algorithm**: A novel algorithm that optimizes per-sample gradient norm computation by fusing norm calculations directly into the backward pass. This eliminates the overhead of separate per-sample gradient materialization, dramatically reducing memory consumption and computational cost.

- **Kernel Operator Fusion**: Advanced kernel-level optimizations that leverage modern GPU architectures:
  - **TMA (Tensor Memory Accelerator)**: Utilizes NVIDIA Hopper's hardware-accelerated memory access patterns for efficient tensor data movement, maximizing memory bandwidth utilization.
  - **T-split**: Optimized tensor splitting strategies that partition computations across thread blocks to maximize parallelism while minimizing synchronization overhead.

- **Ghost Clipping Integration**: Seamless support for Ghost Clipping, enabling per-sample gradient clipping without materializing full per-sample gradients, reducing peak memory usage by up to 10x for large models.

- **FSDP Compatibility**: Native support for PyTorch Fully Sharded Data Parallel (FSDP), enabling privacy-preserving training of large-scale models across multiple GPUs and nodes.

These features collectively deliver **significant acceleration in training throughput** and a **substantial reduction in memory overhead**, making differentially private training practical for large-scale models.

## Installation

**Opacus** utilizes `uv`, an extremely fast Python package and project manager written in Rust. `uv` provides 10-100x faster performance compared to `pip` and offers comprehensive project management with a universal lockfile.

### Quick Start

To set up the environment, execute the following commands:

```bash
# Clone the repository
git clone https://github.com/your-organization/flashnorm.git
cd flashnorm

# Set up the environment (installs uv, creates venv, and installs dependencies)
source setup_env.sh
```

The `setup_env.sh` script performs the following:
1. Installs `uv` package manager
2. Creates a Python virtual environment
3. Activates the virtual environment
4. Installs the package in editable mode with all dependencies

### Manual Installation

If you prefer manual installation:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install the package in editable mode
uv pip install -e .
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (for GPU acceleration)
- NVIDIA Hopper architecture (H100) recommended for TMA optimizations

## Getting Started

Training a model with differential privacy via FlashNorm:

```python
import torch
from torch.optim import SGD
from flashnorm.privacy_engine import FlashNormPrivacyEngine

# Define your model, optimizer, and data loader as usual
model = YourModel()
optimizer = SGD(model.parameters(), lr=0.05)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1024)

# Initialize PrivacyEngine and wrap your components
privacy_engine = FlashNormPrivacyEngine()
model, optimizer, criterion, data_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=data_loader,
    noise_multiplier=1.1,
    max_grad_norm=1.0,
    grad_sample_mode="flash",  # or "ghost", "flash_fuse", etc.
)

# Training proceeds as usual
for batch in data_loader:
    optimizer.zero_grad()
    loss = compute_loss(model, batch)
    loss.backward()
    optimizer.step()
```

### Advanced Usage: Flash / Ghost / Fuse modes

For maximum performance with Flash Norm clipping (non-FSDP):

```python
from flashnorm.grad_sample import GradSampleModuleFastGradientClippingFuse
from flashnorm.optimizers import DPOptimizerFastGradientClipping
from flashnorm.utils.fast_gradient_clipping_utils import DPLossFastGradientClipping

# Wrap model with fused gradient clipping module
model = GradSampleModuleFastGradientClippingFuse(
    model,
    max_grad_norm=1.0,
    use_flash_clipping=True,
    use_ghost_clipping=True,
)

# Use specialized optimizer
optimizer = DPOptimizerFastGradientClipping(
    optimizer=base_optimizer,
    noise_multiplier=1.1,
    max_grad_norm=1.0,
    expected_batch_size=batch_size,
)

# Wrap loss function for proper gradient handling
criterion = DPLossFastGradientClipping(
    module=model,
    optimizer=optimizer,
    criterion=base_criterion,
    loss_reduction="mean",
)
```

## Documentation

For detailed documentation, tutorials, and API references, please visit our [documentation site](https://opacus.ai).

### Tutorials

- [Building an Image Classifier with Differential Privacy](tutorials/building_image_classifier.ipynb)
- [Training a Text Classifier with DP on BERT](tutorials/building_text_classifier.ipynb)
- [Introduction to Advanced Features](tutorials/intro_to_advanced_features.ipynb)

## Benchmarks

Performance benchmarks comparing standard DP-SGD with Flash Norm Clipping optimizations are available in the `benchmarks/` directory.

| Method | Memory Reduction | Speedup |
|--------|-----------------|---------|
| Standard DP-SGD | 1x (baseline) | 1x (baseline) |
| Ghost Clipping | ~5-10x | ~1.5-2x |
| Flash Norm Clipping | ~8-15x | ~2-3x |
| Flash + TMA Fusion | ~10-20x | ~3-5x |

*Results may vary based on model architecture and hardware configuration.*

## Contribution

Please check [Contributing](CONTRIBUTING.md) for more details.

## Code of Conduct

Please check [Code of Conduct](CODE_OF_CONDUCT.md) for more details.

## License

This project is licensed under the Apache-2.0 License.


### Flash Norm Clipping Optimizations

```bibtex
@misc{flashnorm2025,
  author = {Your Name and Collaborators},
  title = {FlashNorm: Fast Differentially Private Training with Ghost and Flash Clipping},
  year = {2025},
  howpublished = {\url{https://github.com/your-organization/opacus}},
}
```

## Acknowledgments

This project builds upon the foundational work of the [Opacus](https://github.com/pytorch/opacus) library by Meta Platforms, Inc. We extend our gratitude to the original authors and the open-source community for their contributions to privacy-preserving machine learning.

