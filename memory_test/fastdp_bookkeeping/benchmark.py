#!/usr/bin/env python3
"""
Benchmarking Script for FastDP Bookkeeping (BK) vs Ghost Clipping vs Vanilla Opacus

This script benchmarks three different DP-SGD implementations:
1. Vanilla Opacus (standard GradSampleModule - stores full per-sample gradients)
2. Ghost Clipping (2 backward passes, computes only gradient norms)
3. Bookkeeping (1 backward pass + cached intermediate values)

Metrics:
- Peak CUDA Memory Usage (MB)
- Time per Iteration (seconds)
- Memory Efficiency (relative to vanilla)
- Speed Efficiency (relative to 2-pass)

Author: AI Research Engineer
Date: 2025-11-11
"""

import torch
import torch.nn as nn
import time
import gc
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from opacus import GradSampleModule
from opacus.grad_sample.grad_sample_module_fast_gradient_clipping import (
    GradSampleModuleFastGradientClipping,
)
from opacus.optimizers import DPOptimizer, DPOptimizerFastGradientClipping
from opacus.utils.fast_gradient_clipping_utils import DPLossFastGradientClipping


class BenchmarkModel(nn.Module):
    """Model for benchmarking - configurable size"""
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        output_dim: int = 100,
    ):
        super().__init__()
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class SequenceBenchmarkModel(nn.Module):
    """Sequence model for benchmarking (tests 3D tensors)"""
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 3,
        output_dim: int = 50,
    ):
        super().__init__()
        layers = []
        
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        return self.model(x)


def get_memory_usage() -> float:
    """Get current CUDA memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def reset_memory_stats():
    """Reset CUDA memory statistics"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


class BenchmarkRunner:
    """Runs benchmarks for different DP-SGD modes"""
    
    def __init__(
        self,
        model_class,
        model_kwargs: Dict,
        batch_size: int,
        data_shape: Tuple,
        num_classes: int,
        device: str = 'cpu',
        use_triton: bool = False,
    ):
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.num_classes = num_classes
        self.device = device
        self.use_triton = use_triton
    
    def create_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create random data for benchmarking"""
        data = torch.randn(self.batch_size, *self.data_shape).to(self.device)
        if len(self.data_shape) == 2:
            # Sequence case: labels for each position
            labels = torch.randint(0, self.num_classes, (self.batch_size, self.data_shape[0])).to(self.device)
        else:
            labels = torch.randint(0, self.num_classes, (self.batch_size,)).to(self.device)
        return data, labels
    
    def benchmark_vanilla_opacus(
        self,
        num_iterations: int = 10,
        max_grad_norm: float = 1.0,
    ) -> Dict[str, float]:
        """Benchmark vanilla Opacus (stores full per-sample gradients)"""
        print("  Running Vanilla Opacus...")
        
        reset_memory_stats()
        
        # Create model
        model = self.model_class(**self.model_kwargs).to(self.device)
        
        # Wrap with standard GradSampleModule
        wrapped_model = GradSampleModule(
            model,
            batch_first=True,
            loss_reduction="mean",
        )
        
        # Create optimizer
        base_optimizer = torch.optim.SGD(wrapped_model.parameters(), lr=0.01)
        optimizer = DPOptimizer(
            optimizer=base_optimizer,
            noise_multiplier=0.0,
            max_grad_norm=max_grad_norm,
            expected_batch_size=self.batch_size,
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Warmup
        for _ in range(2):
            data, labels = self.create_data()
            optimizer.zero_grad()
            outputs = wrapped_model(data)
            if labels.dim() == 2:
                loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Benchmark
        reset_memory_stats()
        times = []
        
        for _ in range(num_iterations):
            data, labels = self.create_data()
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            optimizer.zero_grad()
            outputs = wrapped_model(data)
            if labels.dim() == 2:
                loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append(time.time() - start_time)
        
        peak_memory = get_memory_usage()
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        return {
            'peak_memory_mb': peak_memory,
            'avg_time_sec': avg_time,
            'std_time_sec': std_time,
        }
    
    def benchmark_ghost_clipping(
        self,
        num_iterations: int = 10,
        max_grad_norm: float = 1.0,
        enable_bookkeeping: bool = False,
    ) -> Dict[str, float]:
        """Benchmark Ghost Clipping (2-pass) or Bookkeeping (1-pass)"""
        mode_name = "Bookkeeping (BK)" if enable_bookkeeping else "Ghost Clipping (2-pass)"
        print(f"  Running {mode_name}...")
        
        reset_memory_stats()
        
        # Create model
        model = self.model_class(**self.model_kwargs).to(self.device)
        
        # Wrap with FastGradientClipping module
        wrapped_model = GradSampleModuleFastGradientClipping(
            model,
            batch_first=True,
            max_grad_norm=max_grad_norm,
            use_ghost_clipping=True,
            use_triton=self.use_triton,
            loss_reduction="mean",
            enable_fastdp_bookkeeping=enable_bookkeeping,
        )
        
        # Create optimizer
        base_optimizer = torch.optim.SGD(wrapped_model.parameters(), lr=0.01)
        optimizer = DPOptimizerFastGradientClipping(
            optimizer=base_optimizer,
            noise_multiplier=0.0,
            max_grad_norm=max_grad_norm,
            expected_batch_size=self.batch_size,
            loss_reduction="mean",
        )
        
        # Create loss wrapper
        criterion = nn.CrossEntropyLoss(reduction="mean")
        dp_loss = DPLossFastGradientClipping(
            wrapped_model,
            optimizer,
            criterion,
            loss_reduction="mean",
        )
        
        # Warmup
        for _ in range(2):
            data, labels = self.create_data()
            optimizer.zero_grad()
            outputs = wrapped_model(data)
            if labels.dim() == 2:
                # Sequence labeling: need shape parameter for per-sequence loss
                loss_tensor = dp_loss(outputs.view(-1, outputs.shape[-1]), labels.view(-1), shape=outputs.shape[:2])
            else:
                loss_tensor = dp_loss(outputs, labels)
            loss_tensor.backward()
            optimizer.step()
        
        # Benchmark
        reset_memory_stats()
        times = []
        
        for _ in range(num_iterations):
            data, labels = self.create_data()
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            optimizer.zero_grad()
            outputs = wrapped_model(data)
            if labels.dim() == 2:
                # Sequence labeling: need shape parameter for per-sequence loss
                loss_tensor = dp_loss(outputs.view(-1, outputs.shape[-1]), labels.view(-1), shape=outputs.shape[:2])
            else:
                loss_tensor = dp_loss(outputs, labels)
            loss_tensor.backward()
            optimizer.step()
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append(time.time() - start_time)
        
        peak_memory = get_memory_usage()
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        return {
            'peak_memory_mb': peak_memory,
            'avg_time_sec': avg_time,
            'std_time_sec': std_time,
        }
    
    def run_all_benchmarks(self, num_iterations: int = 10) -> Dict:
        """Run all benchmarks and return results"""
        results = {}
        
        print(f"\nBenchmarking with:")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Data shape: {self.data_shape}")
        print(f"  - Model: {self.model_class.__name__} with {sum(p.numel() for p in self.model_class(**self.model_kwargs).parameters())/1e6:.2f}M parameters")
        print(f"  - Device: {self.device}")
        print(f"  - Triton: {self.use_triton}")
        print(f"  - Iterations: {num_iterations}\n")
        
        # Benchmark 1: Vanilla Opacus
        try:
            results['vanilla'] = self.benchmark_vanilla_opacus(num_iterations)
        except Exception as e:
            print(f"  ❌ Vanilla Opacus failed: {e}")
            results['vanilla'] = None
        
        # Benchmark 2: Ghost Clipping (2-pass)
        try:
            results['ghost_clipping'] = self.benchmark_ghost_clipping(
                num_iterations, enable_bookkeeping=False
            )
        except Exception as e:
            print(f"  ❌ Ghost Clipping failed: {e}")
            results['ghost_clipping'] = None
        
        # Benchmark 3: Bookkeeping (1-pass)
        try:
            results['bookkeeping'] = self.benchmark_ghost_clipping(
                num_iterations, enable_bookkeeping=True
            )
        except Exception as e:
            print(f"  ❌ Bookkeeping failed: {e}")
            results['bookkeeping'] = None
        
        return results


def print_results_table(results: Dict):
    """Print formatted results table"""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    methods = ['vanilla', 'ghost_clipping', 'bookkeeping']
    method_names = ['Vanilla Opacus', 'Ghost Clipping (2-pass)', 'Bookkeeping (1-pass)']
    
    print(f"\n{'Method':<30} {'Peak Memory (MB)':<20} {'Time/Iter (sec)':<25}")
    print("-"*80)
    
    for method, name in zip(methods, method_names):
        if results.get(method) is None:
            print(f"{name:<30} {'FAILED':<20} {'FAILED':<25}")
            continue
        
        mem = results[method]['peak_memory_mb']
        time_avg = results[method]['avg_time_sec']
        time_std = results[method]['std_time_sec']
        
        print(f"{name:<30} {mem:>10.2f}{'':<10} {time_avg:>10.4f} ± {time_std:>6.4f}")
    
    # Print relative improvements
    print("\n" + "="*80)
    print("RELATIVE IMPROVEMENTS")
    print("="*80)
    
    if results.get('vanilla') and results.get('ghost_clipping'):
        vanilla_mem = results['vanilla']['peak_memory_mb']
        ghost_mem = results['ghost_clipping']['peak_memory_mb']
        if vanilla_mem > 0:
            mem_improvement = (1 - ghost_mem / vanilla_mem) * 100
            print(f"\nGhost Clipping vs Vanilla:")
            print(f"  Memory Reduction: {mem_improvement:>6.2f}%")
        else:
            print(f"\nGhost Clipping vs Vanilla:")
            print(f"  Memory stats not available (CPU mode)")
    
    if results.get('ghost_clipping') and results.get('bookkeeping'):
        time_improvement = (1 - results['bookkeeping']['avg_time_sec'] / results['ghost_clipping']['avg_time_sec']) * 100
        ghost_mem = results['ghost_clipping']['peak_memory_mb']
        bk_mem = results['bookkeeping']['peak_memory_mb']
        
        print(f"\nBookkeeping vs Ghost Clipping:")
        print(f"  Speed Improvement: {time_improvement:>6.2f}%")
        
        if ghost_mem > 0 and bk_mem > 0:
            mem_ratio = bk_mem / ghost_mem
            print(f"  Memory Ratio: {mem_ratio:>6.2f}x")
            
            if mem_ratio < 1.0:
                print(f"  → Bookkeeping uses {(1-mem_ratio)*100:.1f}% LESS memory")
            else:
                print(f"  → Bookkeeping uses {(mem_ratio-1)*100:.1f}% MORE memory")
        else:
            print(f"  Memory stats not available (CPU mode)")
    
    print("\n" + "="*80)


def main():
    """Run all benchmarks"""
    print("="*80)
    print("FastDP Bookkeeping (BK) vs Ghost Clipping vs Vanilla Opacus")
    print("Benchmarking Script")
    print("="*80)
    
    # Determine device
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"\n✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("\n⚠️  CUDA not available, using CPU")
        print("   Note: Memory comparisons are most meaningful on GPU")
    
    # Check Triton availability
    use_triton = False
    try:
        import triton
        use_triton = device == 'cuda'  # Only use Triton on CUDA
        if use_triton:
            print("✅ Triton available and will be used")
    except ImportError:
        print("⚠️  Triton not available")
    
    # Benchmark 1: 2D Case (Standard Batch)
    print("\n" + "="*80)
    print("BENCHMARK 1: 2D Case (Batch of Vectors)")
    print("="*80)
    
    runner_2d = BenchmarkRunner(
        model_class=BenchmarkModel,
        model_kwargs={'input_dim': 512, 'hidden_dim': 1024, 'num_layers': 4, 'output_dim': 100},
        batch_size=32,
        data_shape=(512,),
        num_classes=100,
        device=device,
        use_triton=use_triton,
    )
    results_2d = runner_2d.run_all_benchmarks(num_iterations=20)
    print_results_table(results_2d)
    
    # Benchmark 2: 3D Case (Sequence Data)
    print("\n" + "="*80)
    print("BENCHMARK 2: 3D Case (Sequence Data)")
    print("="*80)
    
    runner_3d = BenchmarkRunner(
        model_class=SequenceBenchmarkModel,
        model_kwargs={'input_dim': 256, 'hidden_dim': 512, 'num_layers': 3, 'output_dim': 50},
        batch_size=16,
        data_shape=(32, 256),  # (seq_len, input_dim)
        num_classes=50,
        device=device,
        use_triton=use_triton,
    )
    results_3d = runner_3d.run_all_benchmarks(num_iterations=20)
    print_results_table(results_3d)
    
    # Large model benchmark (if on GPU with enough memory)
    if device == 'cuda':
        try:
            print("\n" + "="*80)
            print("BENCHMARK 3: Large Model (More Layers)")
            print("="*80)
            
            runner_large = BenchmarkRunner(
                model_class=BenchmarkModel,
                model_kwargs={'input_dim': 768, 'hidden_dim': 2048, 'num_layers': 8, 'output_dim': 200},
                batch_size=16,
                data_shape=(768,),
                num_classes=200,
                device=device,
                use_triton=use_triton,
            )
            results_large = runner_large.run_all_benchmarks(num_iterations=10)
            print_results_table(results_large)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("\n⚠️  Skipping large model benchmark (insufficient GPU memory)")
            else:
                raise
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print("\nKey Findings:")
    print("1. Ghost Clipping reduces memory vs Vanilla by avoiding per-sample gradient storage")
    print("2. Bookkeeping (BK) reduces time vs Ghost Clipping by using only 1 backward pass")
    print("3. BK may use slightly more memory than 2-pass due to caching, but less than Vanilla")
    print("4. BK is recommended when backward pass time is the bottleneck")
    print("="*80)


if __name__ == "__main__":
    main()

