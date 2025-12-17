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

### Quick Start

To set up the environment, execute the following commands:

```bash
# Clone the repository
git clone --recursive https://github.com/tiktok-privacy-innovation/PrivacyGo/flashnorm.git
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
- PyTorch 2.8+
- CUDA 12.0+ (for GPU acceleration)
- NVIDIA Hopper architecture (H100) recommended for TMA optimizations

## Getting Started

Training a model with differential privacy via FlashNorm is straightforward. 
You can find a complete, runnable example in [`/opt/tiger/flashnorm/examples/quick_start.py`](/opt/tiger/flashnorm/examples/quick_start.py).



## Benchmarks

Performance benchmarks comparing standard DP-SGD with Flash Norm Clipping optimizations are available in the `plots/` directory.

**Linear Layer**

| Memory | Time |
|:------:|:----:|
| <img src="plots/results_linear/d1024_n8/visualizations/memory_vs_seq_length.png" width="420"/> | <img src="plots/results_linear/d1024_n8/visualizations/time_vs_seq_length.png" width="420"/> |


We benchmark FlashNorm against standard differentially private (DP) normalization methods, including Standard Ghost Clipping and Ghost Clipping with Bookkeeping, using an eight-layer linear model (D = 1024, P = 1024), batch size B = 1, and sequence lengths from 4k to 262k tokens, reporting memory and runtime as ratios relative to a non-DP single-pass baseline. FlashNorm exhibits consistently low and stable overhead across all sequence lengths, with memory usage remaining close to 1.1–1.3× and runtime around 1.5×, even at 262k tokens. In contrast, Standard Ghost Clipping scales poorly: memory overhead grows rapidly (≈2.3× at 8k, ≈3.8× at 16k, and >7× at 32k), and runtime overhead increases sharply (≈5× at 8k, ≈16× at 16k, and ≈37× at 32k), making longer sequences impractical. These results show that FlashNorm avoids the activation materialization and sequence-length–dependent costs of standard DP approaches, enabling practical and predictable DP training for long-context models.

**Transformer**


| Memory | Time |
|:------:|:----:|
| <img src="plots/results_transformer/gpt_2k/visualizations/memory_vs_seq_length.png" width="420"/> | <img src="plots/results_transformer/gpt_2k/visualizations/time_vs_seq_length.png" width="420"/> |

We further evaluate FlashNorm on a GPT-2–style transformer (VOCAB_SIZE = 50,257, hidden size = 768, 12 layers, 12 attention heads) with batch size 1 and sequence lengths from 4k to 32k tokens, reporting memory and runtime as ratios relative to a non-DP single-pass baseline. As shown in the figures, FlashNorm and FlashNorm with bookkeeping (FlashNorm BK) consistently exhibit low and decreasing overhead as sequence length increases: memory usage stays close to the baseline (≈1.02–1.12×), while runtime overhead drops from about 2.1× to 1.7× for FlashNorm and from about 1.4× to nearly 1.05× for FlashNorm BK. In contrast, standard ghost clipping methods scale significantly worse, with memory overhead rising to ~1.23× and runtime overhead exceeding 4× at 32k tokens. These results demonstrate that FlashNorm generalizes beyond linear benchmarks to full transformer models, delivering near–non-DP efficiency for long-context GPT-2 training while substantially outperforming standard DP normalization approaches in both memory and runtime.

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