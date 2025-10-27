# Single GPU Memory Profiler for Opacus

This directory contains tools for profiling GPU memory usage during differential privacy training with Opacus on a single GPU. The profiler is designed to help understand memory consumption patterns and optimize training configurations.

## Files

- `single_gpu_memory_profiler.py` - Main memory profiling script
- `run_single_gpu_analysis.sh` - Automated script to run multiple profiling configurations
- `requirements.txt` - Python dependencies
- `README.md` - This documentation

## Features

- **Single GPU Focus**: Simplified profiler designed specifically for single GPU setups
- **Memory Tracking**: Detailed tracking of GPU memory allocation throughout training
- **LoRA Support**: Optional LoRA (Low-Rank Adaptation) integration for memory-efficient fine-tuning
- **Batch Memory Management**: Support for logical vs physical batch size optimization
- **Visualization**: Automatic generation of memory usage charts and comparisons
- **Multiple Configurations**: Easy testing of different sequence lengths, batch sizes, and LoRA settings

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Opacus** (if not already installed):
   ```bash
   # From the root opacus directory
   pip install -e ../../
   ```

3. **GPU Requirements**:
   - CUDA-compatible GPU
   - Sufficient GPU memory (recommended: 8GB+ for Llama-3.2-3B-Instruct)

## Usage

### Quick Start

Run the automated analysis script:

```bash
./run_single_gpu_analysis.sh
```

For gated models (like Llama), provide your Hugging Face token:

```bash
./run_single_gpu_analysis.sh --token YOUR_HF_TOKEN
```

### Manual Profiling

Run specific configurations manually:

```bash
# Basic profiling with default settings
python single_gpu_memory_profiler.py

# Custom configuration
python single_gpu_memory_profiler.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --seq_length 256 \
    --batch_size 16 \
    --max_physical_batch_size 2 \
    --use_lora \
    --lora_rank 32 \
    --token YOUR_HF_TOKEN
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--token` | None | Hugging Face token for gated models |
| `--model_name` | `meta-llama/Llama-3.2-3B-Instruct` | Model to profile |
| `--seq_length` | 128 | Input sequence length |
| `--batch_size` | 32 | Logical batch size |
| `--max_physical_batch_size` | 1 | Maximum physical batch size |
| `--learning_rate` | 1e-5 | Learning rate |
| `--sigma` | 1.0 | Noise multiplier for DP |
| `--max_grad_norm` | 1.0 | Maximum gradient norm |
| `--num_steps` | 5 | Number of training steps |
| `--results_dir` | `./single_gpu_memory_results` | Output directory |
| `--use_lora` | False | Enable LoRA adaptation |
| `--lora_rank` | 16 | LoRA rank (if enabled) |

## Default Model

The profiler uses **Llama-3.2-3B-Instruct** as the default model <mcreference link="https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct" index="0">0</mcreference>. This model provides a good balance between:
- **Size**: 3B parameters, manageable on most modern GPUs
- **Performance**: Strong performance for sequence classification tasks
- **Availability**: Publicly available through Hugging Face

### Model Requirements

- **GPU Memory**: ~6-8GB for basic inference, 10-12GB for training
- **Access**: May require Hugging Face token for gated access
- **License**: Llama 3.2 Community License

## Output

The profiler generates several types of output:

### 1. Console Output
Real-time memory usage and configuration details during profiling.

### 2. JSON Results (`memory_profile_results.json`)
Detailed profiling data including:
- Memory snapshots at each stage
- Peak memory usage
- Memory breakdown by training phase
- Execution timing

### 3. CSV Summary (`memory_profile_summary.csv`)
Tabular summary of key metrics for easy analysis.

### 4. Visualizations
- `memory_usage_timeline.png`: Memory usage over time
- `peak_memory_comparison.png`: Peak memory comparison across configurations

## Example Configurations

The automated script tests these configurations:

1. **Basic Configuration**
   - Sequence length: 128
   - Batch size: 8
   - No LoRA

2. **Sequence Length Variations**
   - 64, 128, 256 tokens
   - Fixed batch size: 8

3. **Batch Size Variations**
   - 8, 16, 32 samples
   - Fixed sequence length: 128

4. **LoRA Configurations**
   - LoRA rank: 16, 32
   - Comparison with full fine-tuning

5. **Physical Batch Size Tests**
   - 1, 2, 4 physical batches
   - Memory vs speed trade-offs

## Memory Optimization Tips

1. **Use LoRA**: Reduces memory usage by ~50-70% with minimal performance impact
2. **Adjust Physical Batch Size**: Balance memory usage and training speed
3. **Sequence Length**: Shorter sequences use significantly less memory
4. **Gradient Accumulation**: Use logical batch size > physical batch size for memory efficiency

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `max_physical_batch_size`
   - Use shorter `seq_length`
   - Enable `--use_lora`

2. **Model Access Denied**
   - Provide valid Hugging Face token with `--token`
   - Ensure you have access to the model

3. **Import Errors**
   - Install all requirements: `pip install -r requirements.txt`
   - Ensure Opacus is installed: `pip install -e ../../`

### Performance Tips

- **Warm-up**: First run may be slower due to model downloading
- **Memory Fragmentation**: Restart Python between large configuration changes
- **Monitoring**: Use `nvidia-smi` to monitor GPU usage during profiling

## Extending the Profiler

The profiler can be extended to support:

- **Different Models**: Modify `model_name` parameter
- **Custom Datasets**: Replace SNLI dataset in `prepare_snli_dataset()`
- **Additional Metrics**: Add custom memory tracking in `GPUMemoryProfiler`
- **Different Tasks**: Adapt for other NLP tasks beyond sequence classification

## Related Tools

- **FSDP Profiler**: `../fsdp_profile/` for multi-GPU distributed training
- **Opacus Examples**: `../` for other differential privacy examples
- **Benchmarks**: `../../benchmarks/` for performance benchmarking

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review Opacus documentation
3. Open an issue in the Opacus repository