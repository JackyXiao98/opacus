#!/usr/bin/env python3
"""
Benchmark script to compare three implementations:
1. Original Opacus algorithm (opacus.grad_sample.linear)
2. Flash clipping linear implementation
3. Triton-accelerated flash clipping linear
"""

import sys
import os
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple

# Add parent directory to path to import flash_clipping_linear
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add opacus path for original algorithm
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))

from flash_clipping_linear import compute_linear_norm_sample
from triton_version.triton_flash_clipping_linear import compute_linear_norm_sample_triton
from opacus.grad_sample.linear import compute_linear_norm_sample as compute_linear_norm_sample_opacus


def create_test_data(batch_size: int, seq_len: int, input_dim: int, output_dim: int, 
                    device: str = 'cuda') -> Tuple[nn.Linear, List[torch.Tensor], torch.Tensor]:
    """Create test data for benchmarking."""
    # Create linear layer
    layer = nn.Linear(input_dim, output_dim, bias=True).to(device)
    
    # Create activations (3D for sequence data)
    activations = [torch.randn(batch_size, seq_len, input_dim, device=device)]
    
    # Create backprops (3D for sequence data)
    backprops = torch.randn(batch_size, seq_len, output_dim, device=device)
    
    return layer, activations, backprops


def benchmark_function(func, *args, num_warmup: int = 5, num_runs: int = 20) -> float:
    """Benchmark a function and return average execution time."""
    # Warmup runs
    for _ in range(num_warmup):
        _ = func(*args)
        torch.cuda.synchronize()
    
    # Actual timing runs
    start_time = time.time()
    for _ in range(num_runs):
        _ = func(*args)
        torch.cuda.synchronize()
    end_time = time.time()
    
    return (end_time - start_time) / num_runs


def compare_accuracy_three_way(result_opacus: Dict[nn.Parameter, torch.Tensor],
                             result_flash: Dict[nn.Parameter, torch.Tensor], 
                             result_triton: Dict[nn.Parameter, torch.Tensor], 
                             rtol: float = 1e-4, atol: float = 1e-6) -> Dict[str, Dict[str, bool]]:
    """Compare accuracy between three results: Opacus, Flash, and Triton."""
    accuracy = {}
    
    for param_name in ['weight', 'bias']:
        param_opacus = None
        param_flash = None
        param_triton = None
        
        # Find corresponding parameters
        for param, tensor in result_opacus.items():
            if param_name in str(param):
                param_opacus = tensor
                break
        
        for param, tensor in result_flash.items():
            if param_name in str(param):
                param_flash = tensor
                break
                
        for param, tensor in result_triton.items():
            if param_name in str(param):
                param_triton = tensor
                break
        
        accuracy[param_name] = {}
        
        if param_opacus is not None and param_flash is not None and param_triton is not None:
            # Compare Flash vs Opacus
            accuracy[param_name]['flash_vs_opacus'] = torch.allclose(param_flash, param_opacus, rtol=rtol, atol=atol)
            if not accuracy[param_name]['flash_vs_opacus']:
                max_diff = torch.max(torch.abs(param_flash - param_opacus)).item()
                rel_diff = max_diff / torch.max(torch.abs(param_opacus)).item()
                print(f"  {param_name} Flash vs Opacus mismatch: max_abs_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}")
            
            # Compare Triton vs Opacus
            accuracy[param_name]['triton_vs_opacus'] = torch.allclose(param_triton, param_opacus, rtol=rtol, atol=atol)
            if not accuracy[param_name]['triton_vs_opacus']:
                max_diff = torch.max(torch.abs(param_triton - param_opacus)).item()
                rel_diff = max_diff / torch.max(torch.abs(param_opacus)).item()
                print(f"  {param_name} Triton vs Opacus mismatch: max_abs_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}")
            
            # Compare Triton vs Flash
            accuracy[param_name]['triton_vs_flash'] = torch.allclose(param_triton, param_flash, rtol=rtol, atol=atol)
            if not accuracy[param_name]['triton_vs_flash']:
                max_diff = torch.max(torch.abs(param_triton - param_flash)).item()
                rel_diff = max_diff / torch.max(torch.abs(param_flash)).item()
                print(f"  {param_name} Triton vs Flash mismatch: max_abs_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}")
        else:
            # Handle cases where parameters might be None
            accuracy[param_name]['flash_vs_opacus'] = param_flash is None and param_opacus is None
            accuracy[param_name]['triton_vs_opacus'] = param_triton is None and param_opacus is None
            accuracy[param_name]['triton_vs_flash'] = param_triton is None and param_flash is None
    
    return accuracy


def compare_accuracy(result1: Dict[nn.Parameter, torch.Tensor], 
                    result2: Dict[nn.Parameter, torch.Tensor], 
                    rtol: float = 1e-4, atol: float = 1e-6) -> Dict[str, bool]:
    """Compare accuracy between two results (legacy function for compatibility)."""
    accuracy = {}
    
    for param_name in ['weight', 'bias']:
        param1 = None
        param2 = None
        
        # Find corresponding parameters
        for param, tensor in result1.items():
            if param_name in str(param):
                param1 = tensor
                break
        
        for param, tensor in result2.items():
            if param_name in str(param):
                param2 = tensor
                break
        
        if param1 is not None and param2 is not None:
            accuracy[param_name] = torch.allclose(param1, param2, rtol=rtol, atol=atol)
            if not accuracy[param_name]:
                max_diff = torch.max(torch.abs(param1 - param2)).item()
                rel_diff = max_diff / torch.max(torch.abs(param1)).item()
                print(f"  {param_name} mismatch: max_abs_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}")
        else:
            accuracy[param_name] = param1 is None and param2 is None
    
    return accuracy


def run_benchmark_suite():
    """Run comprehensive benchmark suite."""
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmarks.")
        return
    
    device = 'cuda'
    print(f"Running benchmarks on {device}")
    print("=" * 80)
    
    # Test configurations: (batch_size, seq_len, input_dim, output_dim)
    # test_configs = [
    #     (8, 256, 512, 512),      # Small
    #     (8, 512, 512, 512),     # Medium
    #     (8, 1024, 512, 512),    # Large
    #     (8, 2048, 512, 512),  
    #     (8, 4086, 512, 512),  
    #     (8, 8192, 512, 512),  
    #     (8, 16394, 512, 512),    
    # ]

    test_configs = [
        (8, 8192, 128, 128),    
    ]
    
    results = []
    
    for batch_size, seq_len, input_dim, output_dim in test_configs:
        print(f"\nTest Config: B={batch_size}, T={seq_len}, d_in={input_dim}, d_out={output_dim}")
        print("-" * 60)
        
        try:
            # Create test data
            layer, activations, backprops = create_test_data(
                batch_size, seq_len, input_dim, output_dim, device
            )
            
            # Test accuracy first - compare all three algorithms
            print("Testing accuracy...")
            result_opacus = compute_linear_norm_sample_opacus(layer, activations, backprops)
            result_flash = compute_linear_norm_sample(layer, activations, backprops)
            result_triton = compute_linear_norm_sample_triton(layer, activations, backprops)
            
            accuracy = compare_accuracy_three_way(result_opacus, result_flash, result_triton)
            
            # Check if all comparisons pass
            flash_vs_opacus_passed = all(accuracy[param]['flash_vs_opacus'] for param in accuracy)
            triton_vs_opacus_passed = all(accuracy[param]['triton_vs_opacus'] for param in accuracy)
            triton_vs_flash_passed = all(accuracy[param]['triton_vs_flash'] for param in accuracy)
            
            print(f"  Flash vs Opacus - Weight: {'✓' if accuracy.get('weight', {}).get('flash_vs_opacus', False) else '✗'}, "
                  f"Bias: {'✓' if accuracy.get('bias', {}).get('flash_vs_opacus', False) else '✗'}")
            print(f"  Triton vs Opacus - Weight: {'✓' if accuracy.get('weight', {}).get('triton_vs_opacus', False) else '✗'}, "
                  f"Bias: {'✓' if accuracy.get('bias', {}).get('triton_vs_opacus', False) else '✗'}")
            print(f"  Triton vs Flash - Weight: {'✓' if accuracy.get('weight', {}).get('triton_vs_flash', False) else '✗'}, "
                  f"Bias: {'✓' if accuracy.get('bias', {}).get('triton_vs_flash', False) else '✗'}")
            
            overall_accuracy_passed = flash_vs_opacus_passed and triton_vs_opacus_passed and triton_vs_flash_passed
            print(f"  Overall accuracy: {'✓' if overall_accuracy_passed else '✗'}")
            
            if overall_accuracy_passed:
                # Benchmark speed for all three algorithms
                print("Benchmarking speed...")
                
                time_opacus = benchmark_function(
                    compute_linear_norm_sample_opacus, layer, activations, backprops
                )
                
                time_flash = benchmark_function(
                    compute_linear_norm_sample, layer, activations, backprops
                )
                
                time_triton = benchmark_function(
                    compute_linear_norm_sample_triton, layer, activations, backprops
                )
                
                speedup_flash_vs_opacus = time_opacus / time_flash if time_flash > 0 else float('inf')
                speedup_triton_vs_opacus = time_opacus / time_triton if time_triton > 0 else float('inf')
                speedup_triton_vs_flash = time_flash / time_triton if time_triton > 0 else float('inf')
                
                print(f"  Opacus time: {time_opacus*1000:.2f} ms")
                print(f"  Flash time: {time_flash*1000:.2f} ms")
                print(f"  Triton time: {time_triton*1000:.2f} ms")
                print(f"  Flash vs Opacus speedup: {speedup_flash_vs_opacus:.2f}x")
                print(f"  Triton vs Opacus speedup: {speedup_triton_vs_opacus:.2f}x")
                print(f"  Triton vs Flash speedup: {speedup_triton_vs_flash:.2f}x")
                
                results.append({
                    'config': (batch_size, seq_len, input_dim, output_dim),
                    'accuracy_passed': overall_accuracy_passed,
                    'time_opacus': time_opacus,
                    'time_flash': time_flash,
                    'time_triton': time_triton,
                    'speedup_flash_vs_opacus': speedup_flash_vs_opacus,
                    'speedup_triton_vs_opacus': speedup_triton_vs_opacus,
                    'speedup_triton_vs_flash': speedup_triton_vs_flash
                })
            else:
                print("  Skipping speed benchmark due to accuracy issues.")
                results.append({
                    'config': (batch_size, seq_len, input_dim, output_dim),
                    'accuracy_passed': overall_accuracy_passed,
                    'time_opacus': None,
                    'time_flash': None,
                    'time_triton': None,
                    'speedup_flash_vs_opacus': None,
                    'speedup_triton_vs_opacus': None,
                    'speedup_triton_vs_flash': None
                })
                
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'config': (batch_size, seq_len, input_dim, output_dim),
                'accuracy_passed': False,
                'time_opacus': None,
                'time_flash': None,
                'time_triton': None,
                'speedup_flash_vs_opacus': None,
                'speedup_triton_vs_opacus': None,
                'speedup_triton_vs_flash': None,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 120)
    print("BENCHMARK SUMMARY - THREE-WAY COMPARISON")
    print("=" * 120)
    
    print(f"{'Config':<20} {'Acc':<5} {'Opacus(ms)':<12} {'Flash(ms)':<12} {'Triton(ms)':<12} {'F/O':<8} {'T/O':<8} {'T/F':<8}")
    print("-" * 120)
    
    flash_vs_opacus_speedups = []
    triton_vs_opacus_speedups = []
    triton_vs_flash_speedups = []
    accuracy_count = 0
    
    for result in results:
        config_str = f"{result['config'][0]}x{result['config'][1]}x{result['config'][2]}x{result['config'][3]}"
        accuracy_str = "✓" if result['accuracy_passed'] else "✗"
        
        if (result.get('time_opacus') is not None and 
            result.get('time_flash') is not None and 
            result.get('time_triton') is not None):
            
            opacus_str = f"{result['time_opacus']*1000:.2f}"
            flash_str = f"{result['time_flash']*1000:.2f}"
            triton_str = f"{result['time_triton']*1000:.2f}"
            
            flash_opacus_str = f"{result['speedup_flash_vs_opacus']:.2f}x"
            triton_opacus_str = f"{result['speedup_triton_vs_opacus']:.2f}x"
            triton_flash_str = f"{result['speedup_triton_vs_flash']:.2f}x"
            
            flash_vs_opacus_speedups.append(result['speedup_flash_vs_opacus'])
            triton_vs_opacus_speedups.append(result['speedup_triton_vs_opacus'])
            triton_vs_flash_speedups.append(result['speedup_triton_vs_flash'])
        else:
            opacus_str = "N/A"
            flash_str = "N/A"
            triton_str = "N/A"
            flash_opacus_str = "N/A"
            triton_opacus_str = "N/A"
            triton_flash_str = "N/A"
        
        if result['accuracy_passed']:
            accuracy_count += 1
        
        print(f"{config_str:<20} {accuracy_str:<5} {opacus_str:<12} {flash_str:<12} {triton_str:<12} "
              f"{flash_opacus_str:<8} {triton_opacus_str:<8} {triton_flash_str:<8}")
    
    print("-" * 120)
    print(f"Accuracy rate: {accuracy_count}/{len(results)} ({accuracy_count/len(results)*100:.1f}%)")
    print()
    
    if flash_vs_opacus_speedups:
        print("SPEEDUP ANALYSIS:")
        print(f"Flash vs Opacus - Avg: {np.mean(flash_vs_opacus_speedups):.2f}x, "
              f"Best: {max(flash_vs_opacus_speedups):.2f}x, "
              f"Worst: {min(flash_vs_opacus_speedups):.2f}x")
        
        print(f"Triton vs Opacus - Avg: {np.mean(triton_vs_opacus_speedups):.2f}x, "
              f"Best: {max(triton_vs_opacus_speedups):.2f}x, "
              f"Worst: {min(triton_vs_opacus_speedups):.2f}x")
        
        print(f"Triton vs Flash - Avg: {np.mean(triton_vs_flash_speedups):.2f}x, "
              f"Best: {max(triton_vs_flash_speedups):.2f}x, "
              f"Worst: {min(triton_vs_flash_speedups):.2f}x")
    else:
        print("No successful speed benchmarks.")


def test_2d_case():
    """Test 2D case specifically with three-way comparison."""
    print("\n" + "=" * 80)
    print("TESTING 2D CASE - THREE-WAY COMPARISON")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create 2D test data
    batch_size, input_dim, output_dim = 32, 512, 256
    layer = nn.Linear(input_dim, output_dim, bias=True).to(device)
    activations = [torch.randn(batch_size, input_dim, device=device)]
    backprops = torch.randn(batch_size, output_dim, device=device)
    
    print(f"Config: B={batch_size}, d_in={input_dim}, d_out={output_dim}")
    
    # Test accuracy with all three algorithms
    print("Testing accuracy...")
    result_opacus = compute_linear_norm_sample_opacus(layer, activations, backprops)
    result_flash = compute_linear_norm_sample(layer, activations, backprops)
    result_triton = compute_linear_norm_sample_triton(layer, activations, backprops)
    
    accuracy = compare_accuracy_three_way(result_opacus, result_flash, result_triton)
    
    # Check if all comparisons pass
    flash_vs_opacus_passed = all(accuracy[param]['flash_vs_opacus'] for param in accuracy)
    triton_vs_opacus_passed = all(accuracy[param]['triton_vs_opacus'] for param in accuracy)
    triton_vs_flash_passed = all(accuracy[param]['triton_vs_flash'] for param in accuracy)
    
    print(f"Flash vs Opacus - Weight: {'✓' if accuracy.get('weight', {}).get('flash_vs_opacus', False) else '✗'}, "
          f"Bias: {'✓' if accuracy.get('bias', {}).get('flash_vs_opacus', False) else '✗'}")
    print(f"Triton vs Opacus - Weight: {'✓' if accuracy.get('weight', {}).get('triton_vs_opacus', False) else '✗'}, "
          f"Bias: {'✓' if accuracy.get('bias', {}).get('triton_vs_opacus', False) else '✗'}")
    print(f"Triton vs Flash - Weight: {'✓' if accuracy.get('weight', {}).get('triton_vs_flash', False) else '✗'}, "
          f"Bias: {'✓' if accuracy.get('bias', {}).get('triton_vs_flash', False) else '✗'}")
    
    overall_accuracy_passed = flash_vs_opacus_passed and triton_vs_opacus_passed and triton_vs_flash_passed
    print(f"Overall accuracy: {'✓' if overall_accuracy_passed else '✗'}")
    
    if torch.cuda.is_available() and overall_accuracy_passed:
        # Benchmark speed for all three algorithms
        print("Benchmarking speed...")
        
        time_opacus = benchmark_function(
            compute_linear_norm_sample_opacus, layer, activations, backprops
        )
        
        time_flash = benchmark_function(
            compute_linear_norm_sample, layer, activations, backprops
        )
        
        time_triton = benchmark_function(
            compute_linear_norm_sample_triton, layer, activations, backprops
        )
        
        speedup_flash_vs_opacus = time_opacus / time_flash if time_flash > 0 else float('inf')
        speedup_triton_vs_opacus = time_opacus / time_triton if time_triton > 0 else float('inf')
        speedup_triton_vs_flash = time_flash / time_triton if time_triton > 0 else float('inf')
        
        print(f"Opacus time: {time_opacus*1000:.2f} ms")
        print(f"Flash time: {time_flash*1000:.2f} ms")
        print(f"Triton time: {time_triton*1000:.2f} ms")
        print(f"Flash vs Opacus speedup: {speedup_flash_vs_opacus:.2f}x")
        print(f"Triton vs Opacus speedup: {speedup_triton_vs_opacus:.2f}x")
        print(f"Triton vs Flash speedup: {speedup_triton_vs_flash:.2f}x")


if __name__ == "__main__":
    print("Triton vs Flash Clipping Linear Benchmark")
    print("=" * 80)
    
    # Test 2D case first
    test_2d_case()
    
    # Run main benchmark suite for 3D cases
    run_benchmark_suite()
    
    print("\nBenchmark completed!")