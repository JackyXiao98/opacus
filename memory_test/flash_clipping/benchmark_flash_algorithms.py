#!/usr/bin/env python3
"""
Enhanced benchmark script to compare flash clipping implementations:
1. Opacus Original (baseline)
2. Input-Length-Linear Algorithm (PyTorch)
3. Input-Length-Linear Algorithm (Triton)
4. Width-Linear Algorithm (PyTorch)
5. Width-Linear Algorithm (Triton)

This script tests accuracy, performance, and memory usage across various configurations
with visualization of results.
"""

import sys
import os
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Import matplotlib for visualization
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Import the flash clipping algorithms
from flash_clipping_algorithms import compute_linear_norm_sample, is_triton_available

# Add parent directory to path to import opacus
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from opacus.grad_sample.linear import compute_linear_norm_sample as compute_opacus


# Create results directory
RESULTS_DIR = Path(__file__).parent / "benchmark_results"
RESULTS_DIR.mkdir(exist_ok=True)


def create_test_data(batch_size: int, seq_len: int, input_dim: int, output_dim: int, 
                    device: str = 'cuda') -> Tuple[nn.Linear, List[torch.Tensor], torch.Tensor]:
    """Create test data for benchmarking."""
    layer = nn.Linear(input_dim, output_dim, bias=True).to(device)
    activations = [torch.randn(batch_size, seq_len, input_dim, device=device)]
    backprops = torch.randn(batch_size, seq_len, output_dim, device=device)
    return layer, activations, backprops


def create_test_data_2d(batch_size: int, input_dim: int, output_dim: int, 
                       device: str = 'cuda') -> Tuple[nn.Linear, List[torch.Tensor], torch.Tensor]:
    """Create 2D test data (no sequence dimension)."""
    layer = nn.Linear(input_dim, output_dim, bias=True).to(device)
    activations = [torch.randn(batch_size, input_dim, device=device)]
    backprops = torch.randn(batch_size, output_dim, device=device)
    return layer, activations, backprops


def benchmark_with_memory(func, *args, num_warmup: int = 5, num_runs: int = 20, **kwargs) -> Tuple[float, float]:
    """Benchmark a function and return average execution time and peak memory usage."""
    device = args[1][0].device  # Get device from activations
    
    # Warmup runs
    for _ in range(num_warmup):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        _ = func(*args, **kwargs)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    # Actual timing and memory measurement runs
    peak_memories = []
    start_time = time.time()
    
    for _ in range(num_runs):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        
        _ = func(*args, **kwargs)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() / (1024**2)  # Convert to MB
            peak_memories.append(peak_mem)
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    avg_memory = np.mean(peak_memories) if peak_memories else 0.0
    
    return avg_time, avg_memory


def compare_accuracy(result_baseline: Dict[nn.Parameter, torch.Tensor],
                    result_test: Dict[nn.Parameter, torch.Tensor],
                    label: str = "Test",
                    rtol: float = 1e-4, 
                    atol: float = 1e-6) -> Dict[str, bool]:
    """Compare accuracy against baseline."""
    accuracy = {}
    
    for param_name in ['weight', 'bias']:
        param_baseline = None
        param_test = None
        
        for param, tensor in result_baseline.items():
            if param_name in str(param):
                param_baseline = tensor
                break
        
        for param, tensor in result_test.items():
            if param_name in str(param):
                param_test = tensor
                break
        
        if param_baseline is not None and param_test is not None:
            accuracy[param_name] = torch.allclose(param_test, param_baseline, rtol=rtol, atol=atol)
            if not accuracy[param_name]:
                max_diff = torch.max(torch.abs(param_test - param_baseline)).item()
                rel_diff = max_diff / (torch.max(torch.abs(param_baseline)).item() + 1e-10)
                print(f"    {param_name} mismatch ({label} vs Opacus): "
                      f"max_abs_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}")
        else:
            accuracy[param_name] = param_test is None and param_baseline is None
    
    return accuracy


def plot_time_comparison(results: List[Dict], save_path: Path):
    """Create time comparison plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Extract data
    configs = [f"T={r['config'][1]}" for r in results if r.get('time_opacus') is not None]
    
    implementations = ['Opacus', 'I-L PyTorch', 'I-L Triton', 'Width PyTorch', 'Width Triton']
    times = {impl: [] for impl in implementations}
    
    for r in results:
        if r.get('time_opacus') is not None:
            times['Opacus'].append(r['time_opacus'] * 1000)
            times['I-L PyTorch'].append(r.get('time_input_length_pt', 0) * 1000)
            times['I-L Triton'].append(r.get('time_input_length_triton', 0) * 1000)
            times['Width PyTorch'].append(r.get('time_width_pt', 0) * 1000)
            times['Width Triton'].append(r.get('time_width_triton', 0) * 1000)
    
    # Plot 1: Bar chart
    x = np.arange(len(configs))
    width = 0.15
    
    for i, impl in enumerate(implementations):
        offset = (i - 2) * width
        ax1.bar(x + offset, times[impl], width, label=impl)
    
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Execution Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup vs Opacus
    for impl in implementations[1:]:  # Skip Opacus
        speedups = [times['Opacus'][i] / times[impl][i] if times[impl][i] > 0 else 0 
                   for i in range(len(configs))]
        ax2.plot(configs, speedups, marker='o', label=impl)
    
    ax2.axhline(y=1.0, color='r', linestyle='--', label='Baseline (Opacus)')
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Speedup vs Opacus')
    ax2.set_title('Speedup Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved time comparison plot to {save_path}")


def plot_memory_comparison(results: List[Dict], save_path: Path):
    """Create memory usage comparison plots."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    configs = [f"T={r['config'][1]}" for r in results if r.get('memory_opacus') is not None]
    
    implementations = ['Opacus', 'I-L PyTorch', 'I-L Triton', 'Width PyTorch', 'Width Triton']
    memories = {impl: [] for impl in implementations}
    
    for r in results:
        if r.get('memory_opacus') is not None:
            memories['Opacus'].append(r['memory_opacus'])
            memories['I-L PyTorch'].append(r.get('memory_input_length_pt', 0))
            memories['I-L Triton'].append(r.get('memory_input_length_triton', 0))
            memories['Width PyTorch'].append(r.get('memory_width_pt', 0))
            memories['Width Triton'].append(r.get('memory_width_triton', 0))
    
    x = np.arange(len(configs))
    width = 0.15
    
    for i, impl in enumerate(implementations):
        offset = (i - 2) * width
        ax.bar(x + offset, memories[impl], width, label=impl)
    
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Peak Memory (MB)')
    ax.set_title('Peak Memory Usage Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved memory comparison plot to {save_path}")


def plot_speedup_heatmap(results: List[Dict], save_path: Path):
    """Create speedup heatmap."""
    configs = [f"B={r['config'][0]},T={r['config'][1]},d={r['config'][2]}" 
               for r in results if r.get('time_opacus') is not None]
    
    implementations = ['I-L PyTorch', 'I-L Triton', 'Width PyTorch', 'Width Triton']
    
    speedup_matrix = []
    for impl in implementations:
        speedups = []
        for r in results:
            if r.get('time_opacus') is not None:
                time_key = {
                    'I-L PyTorch': 'time_input_length_pt',
                    'I-L Triton': 'time_input_length_triton',
                    'Width PyTorch': 'time_width_pt',
                    'Width Triton': 'time_width_triton'
                }[impl]
                
                if r.get(time_key, 0) > 0:
                    speedup = r['time_opacus'] / r[time_key]
                else:
                    speedup = 0
                speedups.append(speedup)
        speedup_matrix.append(speedups)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    im = ax.imshow(speedup_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=3)
    
    ax.set_xticks(np.arange(len(configs)))
    ax.set_yticks(np.arange(len(implementations)))
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_yticklabels(implementations)
    
    # Add text annotations
    for i in range(len(implementations)):
        for j in range(len(configs)):
            text = ax.text(j, i, f'{speedup_matrix[i][j]:.2f}x',
                         ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title('Speedup vs Opacus Baseline')
    plt.colorbar(im, ax=ax, label='Speedup')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved speedup heatmap to {save_path}")


def run_comprehensive_benchmark():
    """Run comprehensive benchmark suite with all implementations."""
    if not torch.cuda.is_available():
        print("CUDA not available. Running on CPU (limited functionality).")
        device = 'cpu'
    else:
        device = 'cuda'
    
    print(f"Running benchmarks on {device}")
    print(f"Triton available: {is_triton_available()}")
    print("=" * 120)
    
    # Test configurations optimized for algorithm comparison
    test_configs = [
        # Favoring Input-Length (long T, moderate d)
        (1, 8192*4, 4096, 11008),


        # (1, 20, 4096, 11008),
        # (1, 196, 768, 3072),
        # (1, 1024, 1536, 6144),
        
        # Favoring Width (short T, large d)

    ]
    
    results = []
    
    for batch_size, seq_len, input_dim, output_dim in test_configs:
        print(f"\nConfig: B={batch_size}, T={seq_len}, d_in={input_dim}, d_out={output_dim}")
        print("-" * 100)
        
        try:
            layer, activations, backprops = create_test_data(
                batch_size, seq_len, input_dim, output_dim, device
            )
            
            result = {
                'config': (batch_size, seq_len, input_dim, output_dim),
                'accuracy_passed': False,
            }
            
            # Test accuracy against Opacus baseline
            print("Testing accuracy against Opacus baseline...")
            result_opacus = compute_opacus(layer, activations, backprops)
            
            result_il_pt = compute_linear_norm_sample(
                layer, activations, backprops, algorithm="input_length", use_triton=False
            )
            result_w_pt = compute_linear_norm_sample(
                layer, activations, backprops, algorithm="width", use_triton=False
            )
            
            acc_il_pt = compare_accuracy(result_opacus, result_il_pt, "I-L PyTorch")
            acc_w_pt = compare_accuracy(result_opacus, result_w_pt, "Width PyTorch")
            
            accuracy_passed = all(acc_il_pt.values()) and all(acc_w_pt.values())
            
            if is_triton_available():
                result_il_triton = compute_linear_norm_sample(
                    layer, activations, backprops, algorithm="input_length", use_triton=True
                )
                result_w_triton = compute_linear_norm_sample(
                    layer, activations, backprops, algorithm="width", use_triton=True
                )
                
                acc_il_triton = compare_accuracy(result_opacus, result_il_triton, "I-L Triton")
                acc_w_triton = compare_accuracy(result_opacus, result_w_triton, "Width Triton")
                
                accuracy_passed = accuracy_passed and all(acc_il_triton.values()) and all(acc_w_triton.values())
            
            result['accuracy_passed'] = accuracy_passed
            print(f"  Accuracy: {'✓ PASSED' if accuracy_passed else '✗ FAILED'}")
            
            if accuracy_passed and device == 'cuda':
                print("Benchmarking performance and memory...")
                
                # Benchmark Opacus
                time_opacus, mem_opacus = benchmark_with_memory(
                    compute_opacus, layer, activations, backprops
                )
                result['time_opacus'] = time_opacus
                result['memory_opacus'] = mem_opacus
                
                # Benchmark Input-Length PyTorch
                time_il_pt, mem_il_pt = benchmark_with_memory(
                    compute_linear_norm_sample, layer, activations, backprops,
                    algorithm="input_length", use_triton=False
                )
                result['time_input_length_pt'] = time_il_pt
                result['memory_input_length_pt'] = mem_il_pt
                
                # Benchmark Width PyTorch
                time_w_pt, mem_w_pt = benchmark_with_memory(
                    compute_linear_norm_sample, layer, activations, backprops,
                    algorithm="width", use_triton=False
                )
                result['time_width_pt'] = time_w_pt
                result['memory_width_pt'] = mem_w_pt
                
                # Benchmark Triton versions if available
                if is_triton_available():
                    time_il_triton, mem_il_triton = benchmark_with_memory(
                        compute_linear_norm_sample, layer, activations, backprops,
                        algorithm="input_length", use_triton=True
                    )
                    result['time_input_length_triton'] = time_il_triton
                    result['memory_input_length_triton'] = mem_il_triton
                    
                    time_w_triton, mem_w_triton = benchmark_with_memory(
                        compute_linear_norm_sample, layer, activations, backprops,
                        algorithm="width", use_triton=True
                    )
                    result['time_width_triton'] = time_w_triton
                    result['memory_width_triton'] = mem_w_triton
                
                # Print results
                print(f"  Opacus:          {time_opacus*1000:.2f} ms, {mem_opacus:.1f} MB")
                print(f"  I-L PyTorch:     {time_il_pt*1000:.2f} ms, {mem_il_pt:.1f} MB "
                      f"({time_opacus/time_il_pt:.2f}x)")
                print(f"  Width PyTorch:   {time_w_pt*1000:.2f} ms, {mem_w_pt:.1f} MB "
                      f"({time_opacus/time_w_pt:.2f}x)")
                
                if is_triton_available():
                    print(f"  I-L Triton:      {time_il_triton*1000:.2f} ms, {mem_il_triton:.1f} MB "
                          f"({time_opacus/time_il_triton:.2f}x)")
                    print(f"  Width Triton:    {time_w_triton*1000:.2f} ms, {mem_w_triton:.1f} MB "
                          f"({time_opacus/time_w_triton:.2f}x)")
            
            results.append(result)
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'config': (batch_size, seq_len, input_dim, output_dim),
                'accuracy_passed': False,
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "=" * 120)
    print("BENCHMARK SUMMARY")
    print("=" * 120)
    
    # Create summary table
    print(f"{'Config':<25} {'Acc':<5} {'Opacus(ms)':<12} {'I-L PT(ms)':<12} {'I-L T(ms)':<11} "
          f"{'W PT(ms)':<11} {'W T(ms)':<10}")
    print("-" * 120)
    
    for result in results:
        config_str = f"{result['config'][0]}x{result['config'][1]}x{result['config'][2]}x{result['config'][3]}"
        acc_str = "✓" if result['accuracy_passed'] else "✗"
        
        if result.get('time_opacus'):
            opacus_str = f"{result['time_opacus']*1000:.2f}"
            il_pt_str = f"{result.get('time_input_length_pt', 0)*1000:.2f}"
            il_t_str = f"{result.get('time_input_length_triton', 0)*1000:.2f}" if is_triton_available() else "N/A"
            w_pt_str = f"{result.get('time_width_pt', 0)*1000:.2f}"
            w_t_str = f"{result.get('time_width_triton', 0)*1000:.2f}" if is_triton_available() else "N/A"
        else:
            opacus_str = il_pt_str = il_t_str = w_pt_str = w_t_str = "N/A"
        
        print(f"{config_str:<25} {acc_str:<5} {opacus_str:<12} {il_pt_str:<12} {il_t_str:<11} "
              f"{w_pt_str:<11} {w_t_str:<10}")
    
    print("-" * 120)
    
    # Generate visualizations
    if device == 'cuda' and any(r.get('time_opacus') for r in results):
        print("\nGenerating visualizations...")
        plot_time_comparison(results, RESULTS_DIR / "time_comparison.png")
        plot_memory_comparison(results, RESULTS_DIR / "memory_comparison.png")
        plot_speedup_heatmap(results, RESULTS_DIR / "speedup_heatmap.png")
        print(f"\nAll plots saved to: {RESULTS_DIR}")
    
    return results


def test_2d_case():
    """Test 2D case."""
    print("\n" + "=" * 100)
    print("TESTING 2D CASE")
    print("=" * 100)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    batch_size, input_dim, output_dim = 32, 512, 256
    layer, activations, backprops = create_test_data_2d(
        batch_size, input_dim, output_dim, device
    )
    
    print(f"Config: B={batch_size}, d_in={input_dim}, d_out={output_dim}")
    
    result_opacus = compute_opacus(layer, activations, backprops)
    result_flash = compute_linear_norm_sample(
        layer, activations, backprops, algorithm="input_length", use_triton=False
    )
    
    accuracy = compare_accuracy(result_opacus, result_flash, "Flash")
    accuracy_passed = all(accuracy.values())
    
    print(f"Accuracy: {'✓ PASSED' if accuracy_passed else '✗ FAILED'}")
    
    if device == 'cuda' and accuracy_passed:
        time_opacus, _ = benchmark_with_memory(compute_opacus, layer, activations, backprops)
        time_flash, _ = benchmark_with_memory(
            compute_linear_norm_sample, layer, activations, backprops,
            algorithm="input_length", use_triton=False
        )
        
        print(f"Opacus: {time_opacus*1000:.2f} ms")
        print(f"Flash:  {time_flash*1000:.2f} ms")
        print("Note: For 2D case, both algorithms use the same simple implementation")


if __name__ == "__main__":
    print("Enhanced Flash Clipping Algorithms Benchmark")
    print("Comparing Opacus vs Flash Clipping implementations")
    print("=" * 120)
    
    # Run comprehensive benchmark
    run_comprehensive_benchmark()
    
    print("\n" + "=" * 120)
    print("Benchmark completed!")
    print("=" * 120)
