#!/usr/bin/env python3
"""
Comprehensive comparison of per-sample gradient norm computation methods for Linear layers.

This script compares two implementations:
1. Ghost Clipping (from opacus/grad_sample/linear.py): Uses full Gram matrices O(T²) memory
2. FC-PathB method (from flash_clipping_linear.py): Uses tiled computation O(T) memory

The comparison includes:
- Memory usage profiling
- Execution time benchmarks
- IO cost analysis
- Accuracy verification
"""

import gc
import time
import tracemalloc
from typing import Dict, List, Tuple
import psutil
import os

import torch
import torch.nn as nn
from opt_einsum import contract
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from flash_clipping_linear import compute_linear_norm_sample
# from triton_version.triton_flash_clipping_linear import compute_linear_norm_sample_triton
from opacus.grad_sample.linear import compute_linear_norm_sample as original_compute_linear_norm_sample
from triton_version.triton_flash_clipping_linear import compute_linear_norm_sample_triton as flash_compute_linear_norm_sample

def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_gpu_memory_usage() -> Tuple[float, float]:
    """Get GPU memory usage in MB (allocated, cached)"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        cached = torch.cuda.memory_reserved() / 1024 / 1024
        return allocated, cached
    return 0.0, 0.0

# # Original implementation from opacus/grad_sample/linear.py
# def original_compute_linear_norm_sample(
#     layer: nn.Linear, activations: List[torch.Tensor], backprops: torch.Tensor
# ) -> Dict[nn.Parameter, torch.Tensor]:
#     """Original implementation using full Gram matrices"""
#     activations = activations[0]
#     activations = activations.to(backprops.dtype)
    
#     ret = {}
    
#     if backprops.dim() == 2:
#         if layer.weight.requires_grad:
#             g = torch.einsum("n...i,n...i->n", backprops, backprops)
#             a = torch.einsum("n...j,n...j->n", activations, activations)
#             ret[layer.weight] = torch.sqrt((g * a).flatten())
#         if layer.bias is not None and layer.bias.requires_grad:
#             ret[layer.bias] = torch.sqrt(
#                 torch.einsum("n...i,n...i->n", backprops, backprops).flatten()
#             )
#     elif backprops.dim() == 3:
#         if layer.weight.requires_grad:
#             # This creates O(T²) memory usage
#             ggT = torch.einsum("nik,njk->nij", backprops, backprops)  # batchwise g g^T
#             aaT = torch.einsum(
#                 "nik,njk->nij", activations, activations
#             )  # batchwise a a^T
#             ga = torch.einsum("n...i,n...i->n", ggT, aaT).clamp(min=0)
#             ret[layer.weight] = torch.sqrt(ga)
#         if layer.bias is not None and layer.bias.requires_grad:
#             ggT = torch.einsum("nik,njk->nij", backprops, backprops)  # batchwise g g^T
#             ret[layer.bias] = torch.sqrt(torch.einsum("n...i->n", ggT).clamp(min=0))
#     return ret

class MemoryProfiler:
    """Context manager for memory profiling"""
    
    def __init__(self, name: str):
        self.name = name
        self.start_memory = 0
        self.peak_memory = 0
        self.start_gpu_alloc = 0
        self.start_gpu_cached = 0
        self.peak_gpu_memory = 0
        
    def __enter__(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        tracemalloc.start()
        self.start_memory = get_memory_usage()
        self.start_gpu_alloc, self.start_gpu_cached = get_gpu_memory_usage()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        current_memory = get_memory_usage()
        current_gpu_alloc, current_gpu_cached = get_gpu_memory_usage()
        
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        self.peak_memory = peak / 1024 / 1024  # Convert to MB
        
        if torch.cuda.is_available():
            self.peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        print(f"\n=== Memory Profile: {self.name} ===")
        print(f"CPU Memory - Start: {self.start_memory:.2f} MB, End: {current_memory:.2f} MB")
        print(f"CPU Memory - Peak increase: {self.peak_memory:.2f} MB")
        if torch.cuda.is_available():
            print(f"GPU Memory - Start: {self.start_gpu_alloc:.2f} MB, End: {current_gpu_alloc:.2f} MB")
            print(f"GPU Memory - Peak: {self.peak_gpu_memory:.2f} MB")

def create_test_data(B: int, T: int, d: int, p: int, device: str = 'cpu', dtype: torch.dtype = torch.float32):
    """Create test data with specified dimensions"""
    print(f"\nCreating test data: B={B}, T={T}, d={d}, p={p}")
    print(f"Device: {device}, dtype: {dtype}")
    
    # Create activations [B, T, d]
    activations = torch.randn(B, T, d, device=device, dtype=dtype)
    
    # Create gradients [B, T, p] 
    gradients = torch.randn(B, T, p, device=device, dtype=dtype)
    
    # Create a dummy linear layer
    layer = nn.Linear(d, p, bias=True).to(device=device, dtype=dtype)
    
    # Calculate expected memory usage
    activation_memory = B * T * d * 4 / 1024 / 1024  # float32 = 4 bytes
    gradient_memory = B * T * p * 4 / 1024 / 1024
    gram_memory_original = B * T * T * 4 / 1024 / 1024 * 2  # Two Gram matrices
    
    print(f"Expected memory usage:")
    print(f"  Activations: {activation_memory:.2f} MB")
    print(f"  Gradients: {gradient_memory:.2f} MB")
    print(f"  Ghost Clipping Gram matrices: {gram_memory_original:.2f} MB")
    print(f"  Total for Ghost Clipping: {activation_memory + gradient_memory + gram_memory_original:.2f} MB")
    
    return activations, gradients, layer

def benchmark_method(method_func, layer, activations, gradients, method_name: str, tile_size: int = 256, num_runs: int = 10):
    """Benchmark a method with memory and time profiling"""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {method_name}")
    print(f"{'='*60}")
    
    times = []
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        
        with MemoryProfiler(f"{method_name} - Run {run + 1}"):
            # GPU synchronization before timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            if 'triton' in method_name.lower():
                result = method_func(layer, [activations], gradients, tile_size=tile_size)
            else:
                result = method_func(layer, [activations], gradients)
            
            # GPU synchronization after computation
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
        execution_time = (end_time - start_time) * 1000  # Convert to ms
        times.append(execution_time)
        print(f"Execution time: {execution_time:.2f} ms")
        
        # Clean up
        del result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    
    print(f"\n{method_name} Summary:")
    print(f"Average execution time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"Min time: {min(times):.2f} ms, Max time: {max(times):.2f} ms")
    
    return avg_time, std_time

def verify_accuracy(layer, activations, gradients, tile_size: int = 256, tolerance: float = 1e-5):
    """Verify that both methods produce the same results"""
    print(f"\n{'='*60}")
    print("Accuracy Verification")
    print(f"{'='*60}")
    
    # Compute results with both methods
    original_result = original_compute_linear_norm_sample(layer, [activations], gradients)
    fc_pathb_result = flash_compute_linear_norm_sample(layer, [activations], gradients, tile_size=tile_size)
    
    # Compare results
    for param_name in ['weight', 'bias']:
        param = getattr(layer, param_name)
        if param is not None and param.requires_grad:
            original_norm = original_result[param]
            fc_pathb_norm = fc_pathb_result[param]
            
            max_diff = torch.max(torch.abs(original_norm - fc_pathb_norm)).item()
            rel_error = (max_diff / torch.max(original_norm).item()) if torch.max(original_norm).item() > 0 else 0
            
            print(f"{param_name} - Max absolute difference: {max_diff:.2e}")
            print(f"{param_name} - Relative error: {rel_error:.2e}")
            
            if max_diff < tolerance:
                print(f"{param_name} - ✓ PASSED (within tolerance {tolerance:.2e})")
            else:
                print(f"{param_name} - ✗ FAILED (exceeds tolerance {tolerance:.2e})")

def run_comprehensive_test():
    """Run comprehensive comparison tests"""
    print("="*80)
    print("COMPREHENSIVE COMPARISON: Ghost Clipping vs FC-PathB Linear Norm Computation")
    print("="*80)
    
    # Test configurations
    test_configs = [
        # (B, T, d, p, description)
        # (16, 512, 128, 128, "Small sequence (512 tokens)"),
        # (16, 2048, 128, 128, "Medium sequence (2K tokens)"),
        # (16, 4096, 128, 128, "Large sequence (4K tokens)"),
        (16, 8192, 128, 128, "Very large sequence (8K tokens)"),
    ]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    results = []
    
    for B, T, d, p, description in test_configs:
        print(f"\n{'='*80}")
        print(f"TEST: {description}")
        print(f"{'='*80}")
        
        try:
            # Create test data
            activations, gradients, layer = create_test_data(B, T, d, p, device=device)
            
            # Verify accuracy first
            verify_accuracy(layer, activations, gradients)
            
            # Benchmark Ghost Clipping method
            original_time, original_std = benchmark_method(
                original_compute_linear_norm_sample, 
                layer, activations, gradients, 
                "Ghost Clipping"
            )
            
            # Benchmark FC-PathB method
            flash_time, flash_std = benchmark_method(
                flash_compute_linear_norm_sample, 
                layer, activations, gradients, 
                "FC-PathB Method"
            )
            
            # Calculate speedup
            speedup = original_time / flash_time if flash_time > 0 else float('inf')
            
            results.append({
                'config': description,
                'B': B, 'T': T, 'd': d, 'p': p,
                'original_time': original_time,
                'flash_time': flash_time,
                'speedup': speedup
            })
            
            print(f"\nSpeedup: {speedup:.2f}x")
            
        except Exception as e:
            print(f"Error in test {description}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY RESULTS")
    print(f"{'='*80}")
    print(f"{'Configuration':<25} {'Ghost Clipping (ms)':<20} {'FC-PathB (ms)':<15} {'Speedup':<10}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['config']:<25} {result['original_time']:<20.2f} {result['flash_time']:<15.2f} {result['speedup']:<10.2f}x")

def create_tile_size_visualizations(sequence_lengths, original_gpu_memory, flash_gpu_memory_by_tile):
    """Create GPU memory usage visualizations for different tile sizes"""
    print(f"\n{'='*60}")
    print("CREATING GPU MEMORY VISUALIZATIONS FOR DIFFERENT TILE SIZES")
    print(f"{'='*60}")
    
    tile_sizes = list(flash_gpu_memory_by_tile.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Different colors for each tile size
    
    # Convert to numpy arrays for easier manipulation
    seq_lengths = np.array(sequence_lengths)
    orig_gpu_mem = np.array(original_gpu_memory)
    
    # Create individual plots for each tile size
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Plot 1: Ghost Clipping Method
    axes[0].plot(seq_lengths, orig_gpu_mem, 'ro-', linewidth=2, markersize=8, label='Ghost Clipping')
    axes[0].set_xlabel('Sequence Length (T)', fontsize=12)
    axes[0].set_ylabel('Peak GPU Memory (MB)', fontsize=12)
    axes[0].set_title('Ghost Clipping: GPU Memory vs Sequence Length\n(O(T²) Memory Complexity)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    
    # Add data point labels for original method
    for i, (x, y) in enumerate(zip(seq_lengths, orig_gpu_mem)):
        axes[0].annotate(f'{y:.1f}MB', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    # Plot 2-6: Flash Method with different tile sizes
    for idx, (tile_size, color) in enumerate(zip(tile_sizes, colors)):
        ax_idx = idx + 1
        if ax_idx < len(axes):
            flash_gpu_mem = np.array(flash_gpu_memory_by_tile[tile_size])
            
            axes[ax_idx].plot(seq_lengths, flash_gpu_mem, 'o-', color=color, linewidth=2, markersize=8,
                         label=f'FC-PathB Method (Tile={tile_size})')
            axes[ax_idx].set_xlabel('Sequence Length (T)', fontsize=12)
            axes[ax_idx].set_ylabel('Peak GPU Memory (MB)', fontsize=12)
            axes[ax_idx].set_title(f'FC-PathB Method (Tile Size={tile_size}): \nGPU Memory vs Sequence Length',
                               fontsize=12, fontweight='bold')
            axes[ax_idx].grid(True, alpha=0.3)
            axes[ax_idx].legend(fontsize=11)
            
            # Add data point labels
            for i, (x, y) in enumerate(zip(seq_lengths, flash_gpu_mem)):
                axes[ax_idx].annotate(f'{y:.1f}MB', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    # Hide unused subplot
    if len(tile_sizes) < 5:
        axes[5].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('gpu_memory_by_tile_size.png', dpi=300, bbox_inches='tight')
    print("✓ Saved tile size comparison plots to: gpu_memory_by_tile_size.png")
    
    # Create a combined comparison plot
    fig2, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Plot Ghost Clipping method
    ax.plot(seq_lengths, orig_gpu_mem, 'ro-', linewidth=3, markersize=10, label='Ghost Clipping (O(T²))')
    
    # Plot flash method with different tile sizes
    for tile_size, color in zip(tile_sizes, colors):
        flash_gpu_mem = np.array(flash_gpu_memory_by_tile[tile_size])
        ax.plot(seq_lengths, flash_gpu_mem, 'o-', color=color, linewidth=2, markersize=8,
                label=f'FC-PathB Method (Tile={tile_size})')
    
    ax.set_xlabel('Sequence Length (T)', fontsize=14)
    ax.set_ylabel('Peak GPU Memory Usage (MB)', fontsize=14)
    ax.set_title('GPU Memory Usage Comparison: Original vs FC-PathB Method (Different Tile Sizes)', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='upper left')
    
    # Set log scale if memory usage varies significantly
    all_memory_values = list(orig_gpu_mem) + [mem for tile_mem in flash_gpu_memory_by_tile.values() for mem in tile_mem]
    if max(all_memory_values) / min(all_memory_values) > 10:
        # ax.set_yscale('log')
        ax.set_ylabel('Peak GPU Memory Usage (MB)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('gpu_memory_combined_tile_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved combined tile size comparison plot to: gpu_memory_combined_tile_comparison.png")
    
    # Close the plots to free memory
    plt.close('all')
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("GPU MEMORY USAGE SUMMARY BY TILE SIZE")
    print(f"{'='*80}")
    
    # Header
    header = f"{'Seq Length':<12} {'Ghost Clipping':<12}"
    for tile_size in tile_sizes:
        header += f"{'Tile=' + str(tile_size):<12}"
    print(header)
    print("-" * len(header))
    
    # Data rows
    for i, T in enumerate(sequence_lengths):
        row = f"{T:<12} {orig_gpu_mem[i]:<12.2f}"
        for tile_size in tile_sizes:
            flash_mem = flash_gpu_memory_by_tile[tile_size][i]
            row += f"{flash_mem:<12.2f}"
        print(row)
    
    # Memory savings summary
    print(f"\n{'='*80}")
    print("MEMORY SAVINGS SUMMARY (Ghost Clipping / FC-PathB)")
    print(f"{'='*80}")
    
    header = f"{'Seq Length':<12}"
    for tile_size in tile_sizes:
        header += f"{'Tile=' + str(tile_size):<12}"
    print(header)
    print("-" * len(header))
    
    for i, T in enumerate(sequence_lengths):
        row = f"{T:<12}"
        for tile_size in tile_sizes:
            flash_mem = flash_gpu_memory_by_tile[tile_size][i]
            savings = orig_gpu_mem[i] / flash_mem if flash_mem > 0 else float('inf')
            row += f"{savings:<12.2f}x"
        print(row)

def run_memory_scaling_test():
    """Test GPU memory scaling with different sequence lengths and tile sizes"""
    print(f"\n{'='*80}")
    print("GPU MEMORY SCALING TEST WITH DIFFERENT TILE SIZES")
    print(f"{'='*80}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Warning: CUDA not available, using CPU. GPU memory measurements will be 0.")
    
    B, d, p = 8, 128, 128
    sequence_lengths = [1024, 2048, 4096, 8192, 16384]  # Reduced for stability
    tile_sizes = [64, 128, 256, 512, 1024]
    
    original_gpu_memory = []
    flash_gpu_memory_by_tile = {tile_size: [] for tile_size in tile_sizes}
    successful_lengths = []
    
    print(f"Testing with B={B}, d={d}, p={p}, device={device}")
    print(f"Tile sizes: {tile_sizes}")
    print(f"Sequence lengths: {sequence_lengths}")
    
    for T in sequence_lengths:
        print(f"\n{'='*60}")
        print(f"Testing sequence length T={T}")
        print(f"{'='*60}")
        
        try:
            activations, gradients, layer = create_test_data(B, T, d, p, device=device)
            
            # Test Ghost Clipping method
            print(f"Testing Ghost Clipping...")
            with MemoryProfiler("Ghost Clipping") as original_prof:
                _ = original_compute_linear_norm_sample(layer, [activations], gradients)
            
            original_gpu_mem = original_prof.peak_gpu_memory
            print(f"Ghost Clipping GPU memory: {original_gpu_mem:.2f} MB")
            
            # Clean up before testing flash methods
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Test flash method with different tile sizes
            tile_results = {}
            for tile_size in tile_sizes:
                print(f"Testing FC-PathB Method with tile_size={tile_size}...")
                
                try:
                    with MemoryProfiler(f"FC-PathB-{tile_size}") as flash_prof:
                        _ = flash_compute_linear_norm_sample(layer, [activations], gradients, tile_size=tile_size)
                    
                    flash_gpu_mem = flash_prof.peak_gpu_memory
                    tile_results[tile_size] = flash_gpu_mem
                    print(f"FC-PathB method (tile={tile_size}) GPU memory: {flash_gpu_mem:.2f} MB")
                    
                    # Clean up between tile size tests
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error with tile_size={tile_size}: {e}")
                    tile_results[tile_size] = 0.0
            
            # Store results if we have at least some successful tile size tests
            if any(mem > 0 for mem in tile_results.values()):
                successful_lengths.append(T)
                original_gpu_memory.append(original_gpu_mem)
                
                for tile_size in tile_sizes:
                    flash_gpu_memory_by_tile[tile_size].append(tile_results.get(tile_size, 0.0))
                
                # Print summary for this sequence length
                print(f"\nSummary for T={T}:")
                print(f"  Ghost Clipping: {original_gpu_mem:.2f} MB")
                for tile_size in tile_sizes:
                    flash_mem = tile_results.get(tile_size, 0.0)
                    savings = original_gpu_mem / flash_mem if flash_mem > 0 else float('inf')
                    print(f"  Tile {tile_size}: {flash_mem:.2f} MB (savings: {savings:.2f}x)")
            
        except Exception as e:
            print(f"T={T}: Error - {e}")
            import traceback
            traceback.print_exc()
    
    # Create visualizations if we have successful results
    if successful_lengths:
        print(f"\nCreating visualizations for {len(successful_lengths)} successful sequence lengths...")
        create_tile_size_visualizations(successful_lengths, original_gpu_memory, flash_gpu_memory_by_tile)
    else:
        print("No successful tests to visualize.")

if __name__ == "__main__":
    print("Starting comprehensive comparison of linear norm computation methods...")
    
    # Run main comparison
    run_comprehensive_test()
    
    # Run memory scaling test
    # run_memory_scaling_test()
    
    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"{'='*80}")