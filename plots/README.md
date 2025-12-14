# FSDP Llama3 Memory and Time Profiling

This directory contains a comprehensive experiment suite to compare memory usage and training time for Llama3-8B-Instruct model with FSDP2 across different DP-SGD training modes and sequence lengths.

## Overview

The experiment compares three training modes:
- **no_dp**: Vanilla FSDP2 training (no differential privacy)
- **ghost_fsdp**: Ghost Clipping DP-SGD with FSDP2
- **flash_fsdp**: Flash Clipping DP-SGD with FSDP2

Each mode is tested across three sequence lengths:
- 1024 tokens
- 2048 tokens
- 4096 tokens

## Prerequisites

1. **CUDA GPUs**: Multi-GPU setup required (FSDP2 requires at least 2 GPUs)
2. **HuggingFace Token**: Access to gated models (Llama3)
3. **Python Environment**: Python 3.8+ with the following packages:
   - PyTorch with CUDA support
   - transformers
   - opacus
   - matplotlib
   - numpy
   - datasets

## Setup

1. Set your HuggingFace token as an environment variable:
```bash
export HF_TOKEN=your_huggingface_token_here
```

2. Ensure you have activated the virtual environment:
```bash
source .venv/bin/activate
```

3. Ensure you have accepted the Llama3 model license on HuggingFace:
   - Visit: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
   - Accept the license agreement

## Running Experiments

### Run All Experiments

To run the complete experiment suite (9 experiments: 3 modes × 3 sequence lengths):

```bash
cd memory_test/fsdp_llama3_profiling
bash run_all_experiments.sh
```

This will:
1. Run each experiment in an isolated Python process
2. Save results to `results/run_YYYYMMDD_HHMMSS/`
3. Generate visualizations automatically
4. Print a summary table at the end

**Note**: The full experiment suite can take several hours to complete, depending on your GPU configuration.

### Run Single Experiment

To run a single experiment configuration:

```bash
python single_experiment.py \
    --mode ghost_fsdp \
    --seq-length 2048 \
    --batch-size 2 \
    --num-iter 3 \
    --warmup-iter 1 \
    --output results/test_result.json \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --token $HF_TOKEN
```

### Visualization Only

If you already have results and want to regenerate visualizations:

```bash
python visualize_results.py \
    --input-dir results/run_YYYYMMDD_HHMMSS \
    --output-dir results/run_YYYYMMDD_HHMMSS/visualizations
```

## Configuration

You can modify the experiment parameters in `run_all_experiments.sh`:

```bash
# Model configuration
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
BATCH_SIZE=2              # Batch size per GPU
NUM_ITER=3                # Number of profiling iterations
WARMUP_ITER=1             # Number of warmup iterations

# DP-SGD parameters
SIGMA=1.0                 # Noise multiplier
MAX_GRAD_NORM=1.0         # Gradient clipping norm

# Test configurations
SEQ_LENGTHS=(1024 2048 4096)
MODES=("no_dp" "ghost_fsdp" "flash_fsdp")
```

## Output Structure

After running experiments, the output directory structure will be:

```
results/run_YYYYMMDD_HHMMSS/
├── no_dp_seq1024_result.json
├── no_dp_seq2048_result.json
├── no_dp_seq4096_result.json
├── ghost_fsdp_seq1024_result.json
├── ghost_fsdp_seq2048_result.json
├── ghost_fsdp_seq4096_result.json
├── flash_fsdp_seq1024_result.json
├── flash_fsdp_seq2048_result.json
├── flash_fsdp_seq4096_result.json
└── visualizations/
    ├── memory_comparison.png
    ├── time_comparison.png
    ├── memory_vs_time_tradeoff.png
    ├── overhead_analysis.png
    └── summary.txt
```

## Result Files

Each JSON result file contains:

```json
{
  "mode": "ghost_fsdp",
  "seq_length": 2048,
  "batch_size": 2,
  "total_batch_size": 4,
  "num_gpus": 2,
  "peak_memory_mb": 12345.67,
  "peak_memory_gb": 12.35,
  "avg_time_ms": 234.56,
  "config": {
    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
    "vocab_size": 128256,
    "learning_rate": 1e-05,
    "sigma": 1.0,
    "max_grad_norm": 1.0
  }
}
```

## Visualizations

The experiment generates four types of visualizations:

### 1. Memory Comparison
Bar charts showing peak memory usage for each mode, grouped by sequence length.

### 2. Time Comparison
Bar charts showing average iteration time for each mode, grouped by sequence length.

### 3. Memory vs Time Tradeoff
Scatter plot showing the relationship between memory usage and training time across all configurations.

### 4. Overhead Analysis
Two bar charts showing:
- Memory overhead (GB) of DP modes compared to non-DP baseline
- Time overhead (%) of DP modes compared to non-DP baseline

### 5. Summary Table
Text file with detailed metrics for all experiments and overhead calculations.

## Interpreting Results

### Memory Usage
- **no_dp**: Baseline memory without DP-SGD overhead
- **ghost_fsdp**: Memory includes per-sample gradient norms tracking
- **flash_fsdp**: Memory optimized version with fused operations

### Training Time
- **no_dp**: Fastest, no DP-SGD overhead
- **ghost_fsdp**: Additional overhead from ghost clipping
- **flash_fsdp**: Optimized time with Triton kernels (if available)

### Key Metrics
- **Peak Memory (GB)**: Maximum GPU memory allocated during training
- **Avg Time (ms)**: Average time per training iteration
- **Memory Overhead**: Additional memory required by DP-SGD methods
- **Time Overhead**: Additional training time required by DP-SGD methods

## Troubleshooting

### CUDA Out of Memory
If you encounter OOM errors, try:
- Reducing `BATCH_SIZE` in `run_all_experiments.sh`
- Testing with shorter sequence lengths first
- Using gradient checkpointing (requires code modification)

### HuggingFace Authentication
If authentication fails:
- Verify your token: `huggingface-cli whoami`
- Re-login: `huggingface-cli login`
- Ensure you've accepted the Llama3 license

### Multi-GPU Issues
- Ensure all GPUs are visible: `nvidia-smi`
- Check NCCL installation: `python -c "import torch; print(torch.cuda.nccl.version())"`
- Verify distributed setup: `python -c "import torch.distributed as dist"`

### Visualization Errors
If visualization fails:
- Check matplotlib installation: `pip install matplotlib`
- Verify results files exist: `ls results/run_*/`
- Check JSON file integrity: `python -m json.tool <result_file.json>`

## Implementation Details

### Synthetic Data Generation
The experiments use randomly generated data to avoid dataset loading overhead and enable flexible sequence length testing. The model performs sequence classification (predicting one label per sequence):

```python
def generate_synthetic_batch(batch_size, seq_length, vocab_size, device, num_labels=3):
    """Generate synthetic random data batch for sequence classification"""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    labels = torch.randint(0, num_labels, (batch_size,), device=device)  # One label per sequence
    attention_mask = torch.ones((batch_size, seq_length), device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
```

**Note**: We use `AutoModelForSequenceClassification` which outputs one classification label per sequence (not per token), matching the original SNLI classification task in the reference implementation.

### Memory Tracking
Memory is tracked using PyTorch's built-in CUDA memory statistics:

```python
torch.cuda.reset_peak_memory_stats(device)
# ... training loop ...
peak_memory_bytes = torch.cuda.max_memory_allocated(device)
```

### Time Measurement
Time is measured using CUDA events for accurate GPU timing:

```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
# ... iteration ...
end_event.record()
torch.cuda.synchronize()
elapsed_ms = start_event.elapsed_time(end_event)
```

## Citation

If you use this profiling framework in your research, please cite the Opacus library:

```bibtex
@inproceedings{opacus,
  title={Opacus: User-Friendly Differential Privacy Library in PyTorch},
  author={Ashkan Yousefpour and Igor Shilov and Alexandre Sablayrolles and Davide Testuggine and Karthik Prasad and Mani Malek and John Nguyen and Sayan Ghosh and Akash Bharadwaj and Jessica Zhao and Graham Cormode and Ilya Mironov},
  booktitle={arXiv preprint arXiv:2109.12298},
  year={2021}
}
```

## License

This code is part of the Opacus project and is licensed under Apache License 2.0.

## Support

For issues or questions:
- Open an issue on the Opacus GitHub repository
- Check the Opacus documentation: https://opacus.ai/
- Join the PyTorch Slack channel for Opacus discussions

