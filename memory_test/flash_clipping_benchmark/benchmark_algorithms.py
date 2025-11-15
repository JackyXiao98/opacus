"""
Benchmark Flash Clipping Algorithms

This script benchmarks different gradient norm computation algorithms:
1. PyTorch input_length
2. PyTorch width
3. Triton input_length
4. Triton width

Measurements:
- Wall-clock time (median of 10 runs with warmup)
- Peak GPU memory usage
- Memory allocated/reserved
- Numerical accuracy (relative error vs baseline)
"""

import sys
import os
import time
import json
from typing import Dict, List, Tuple, Any
from datetime import datetime

import torch
import torch.nn as nn

# Add opacus to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from opacus.grad_sample.triton_kernels import (
    _input_length_frobenius,
    _width_frobenius,
    _input_length_frobenius_triton,
    _width_frobenius_triton,
    is_triton_available,
)


def benchmark_algorithm(
    algorithm_fn,
    A: torch.Tensor,
    G: torch.Tensor,
    tile_size: int,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> Dict[str, Any]:
    """
    Benchmark a single algorithm configuration.
    
    Args:
        algorithm_fn: Function to benchmark
        A: Activation tensor [B, T, d_a]
        G: Gradient tensor [B, T, d_g]
        tile_size: Tile size parameter
        num_warmup: Number of warmup runs
        num_runs: Number of measurement runs
    
    Returns:
        Dictionary with timing and memory statistics
    """
    device = A.device
    
    # Warmup runs
    for _ in range(num_warmup):
        _ = algorithm_fn(A, G, tile_size=tile_size, dtype_acc=torch.float32)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    
    # Timed runs
    times = []
    for _ in range(num_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        result = algorithm_fn(A, G, tile_size=tile_size, dtype_acc=torch.float32)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        times.append(end - start)
    
    # Memory statistics
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)  # MB
        allocated_memory = torch.cuda.memory_allocated(device) / (1024**2)  # MB
        reserved_memory = torch.cuda.memory_reserved(device) / (1024**2)  # MB
    else:
        peak_memory = 0.0
        allocated_memory = 0.0
        reserved_memory = 0.0
    
    return {
        'times': times,
        'median_time': float(torch.tensor(times).median()),
        'mean_time': float(torch.tensor(times).mean()),
        'std_time': float(torch.tensor(times).std()),
        'min_time': min(times),
        'max_time': max(times),
        'peak_memory_mb': peak_memory,
        'allocated_memory_mb': allocated_memory,
        'reserved_memory_mb': reserved_memory,
        'result': result.cpu(),
    }


def compute_relative_error(result: torch.Tensor, baseline: torch.Tensor) -> float:
    """Compute relative error between result and baseline."""
    abs_diff = torch.abs(result - baseline)
    rel_error = abs_diff / (torch.abs(baseline) + 1e-10)
    return float(rel_error.max())


def run_benchmark_suite(
    shapes: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]],
    tile_sizes: List[int],
    device: str = 'cuda',
) -> Dict[str, Any]:
    """
    Run complete benchmark suite.
    
    Args:
        shapes: List of (A_shape, G_shape) tuples
        tile_sizes: List of tile sizes to test
        device: Device to run on ('cuda' or 'cpu')
    
    Returns:
        Dictionary with all benchmark results
    """
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'device': device,
            'cuda_available': torch.cuda.is_available(),
            'triton_available': is_triton_available(),
            'shapes': shapes,
            'tile_sizes': tile_sizes,
        },
        'benchmarks': []
    }
    
    if device == 'cuda' and torch.cuda.is_available():
        results['metadata']['gpu_name'] = torch.cuda.get_device_name(0)
        results['metadata']['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    # Algorithm configurations
    algorithms = [
        ('pytorch_input_length', _input_length_frobenius),
        ('pytorch_width', _width_frobenius),
        ('triton_input_length', _input_length_frobenius_triton),
        ('triton_width', _width_frobenius_triton),
    ]
    
    for shape_idx, (A_shape, G_shape) in enumerate(shapes):
        print(f"\n{'='*80}")
        print(f"Testing Shape {shape_idx + 1}: A={A_shape}, G={G_shape}")
        print(f"{'='*80}")
        
        # Create test tensors
        if device == 'cuda' and torch.cuda.is_available():
            A = torch.randn(A_shape, device=device, dtype=torch.float32)
            G = torch.randn(G_shape, device=device, dtype=torch.float32)
        else:
            # Use smaller tensors for CPU to avoid OOM
            if A_shape[1] > 1024:
                print(f"Skipping large shape on CPU: {A_shape}")
                continue
            A = torch.randn(A_shape, device='cpu', dtype=torch.float32)
            G = torch.randn(G_shape, device='cpu', dtype=torch.float32)
        
        for tile_size in tile_sizes:
            print(f"\n  Tile Size: {tile_size}")
            print(f"  {'-'*76}")
            
            baseline_result = None
            
            for algo_name, algo_fn in algorithms:
                print(f"    Testing {algo_name}...", end=' ', flush=True)
                
                try:
                    bench_result = benchmark_algorithm(
                        algo_fn, A, G, tile_size,
                        num_warmup=3, num_runs=10
                    )
                    
                    # Compute accuracy vs baseline (first algorithm)
                    if baseline_result is None:
                        baseline_result = bench_result['result']
                        rel_error = 0.0
                    else:
                        rel_error = compute_relative_error(
                            bench_result['result'],
                            baseline_result
                        )
                    
                    result_entry = {
                        'shape_idx': shape_idx,
                        'A_shape': list(A_shape),
                        'G_shape': list(G_shape),
                        'tile_size': tile_size,
                        'algorithm': algo_name,
                        'median_time_sec': bench_result['median_time'],
                        'mean_time_sec': bench_result['mean_time'],
                        'std_time_sec': bench_result['std_time'],
                        'min_time_sec': bench_result['min_time'],
                        'max_time_sec': bench_result['max_time'],
                        'peak_memory_mb': bench_result['peak_memory_mb'],
                        'allocated_memory_mb': bench_result['allocated_memory_mb'],
                        'reserved_memory_mb': bench_result['reserved_memory_mb'],
                        'relative_error': rel_error,
                    }
                    
                    results['benchmarks'].append(result_entry)
                    
                    print(f"✓ {bench_result['median_time']*1000:.2f}ms "
                          f"(mem: {bench_result['peak_memory_mb']:.1f}MB, "
                          f"err: {rel_error:.2e})")
                    
                except Exception as e:
                    print(f"✗ Error: {str(e)}")
                    results['benchmarks'].append({
                        'shape_idx': shape_idx,
                        'A_shape': list(A_shape),
                        'G_shape': list(G_shape),
                        'tile_size': tile_size,
                        'algorithm': algo_name,
                        'error': str(e),
                    })
        
        # Clean up
        del A, G
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def print_summary(results: Dict[str, Any]):
    """Print summary of benchmark results."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    benchmarks = results['benchmarks']
    
    # Group by shape and tile_size
    from collections import defaultdict
    grouped = defaultdict(list)
    
    for b in benchmarks:
        if 'error' not in b:
            key = (tuple(b['A_shape']), tuple(b['G_shape']), b['tile_size'])
            grouped[key].append(b)
    
    for key, group in grouped.items():
        A_shape, G_shape, tile_size = key
        print(f"\nShape: A={A_shape}, G={G_shape}, Tile={tile_size}")
        print(f"{'-'*80}")
        
        # Sort by median time
        group_sorted = sorted(group, key=lambda x: x['median_time_sec'])
        
        fastest_time = group_sorted[0]['median_time_sec']
        
        for b in group_sorted:
            speedup = fastest_time / b['median_time_sec']
            print(f"  {b['algorithm']:25s}: "
                  f"{b['median_time_sec']*1000:7.2f}ms  "
                  f"speedup: {speedup:5.2f}x  "
                  f"mem: {b['peak_memory_mb']:6.1f}MB  "
                  f"err: {b['relative_error']:.2e}")


def main():
    """Main benchmark runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark flash clipping algorithms')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to run benchmarks on')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='Output JSON file for results')
    parser.add_argument('--small', action='store_true',
                        help='Run small test (for quick testing)')
    
    args = parser.parse_args()
    
    # Test shapes based on user's examples
    if args.small:
        shapes = [
            ((2, 1024, 512), (2, 1024, 256)),
        ]
        tile_sizes = [256, 512]
    else:
        shapes = [
            # User's example shapes
            ((2, 16384, 2048), (2, 16384, 512)),
            ((2, 16384, 2048), (2, 16384, 2048)),
        ]
        tile_sizes = [256, 512, 1024, 2048]
    
    print("Flash Clipping Algorithm Benchmark")
    print("="*80)
    print(f"Device: {args.device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Triton Available: {is_triton_available()}")
    
    if args.device == 'cuda' and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Run benchmarks
    results = run_benchmark_suite(shapes, tile_sizes, device=args.device)
    
    # Print summary
    print_summary(results)
    
    # Save results
    output_path = os.path.join(os.path.dirname(__file__), args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")


if __name__ == '__main__':
    main()

