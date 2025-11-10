#!/usr/bin/env python3
"""
Detailed memory profiling for SimpleBigModel to identify bottlenecks.
Profiles memory usage and IO costs per module (attention, FFN, lm_head, etc.)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import gc
from collections import defaultdict
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Import from parent directory
import sys
import os
# Add parent directory to path to import flash_clipping_linear
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add opacus path for original algorithm
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))

from opacus.layers import DPMultiheadAttention
from opacus.grad_sample import GradSampleModuleFastGradientClipping
from opacus.optimizers import DPOptimizerFastGradientClipping
from memory_test.test_algo.compare_algorithms import SimpleBigModel


class DetailedMemoryProfiler:
    """Profiles memory usage per module with hooks"""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.is_cuda = device == "cuda" and torch.cuda.is_available()
        
        # Store memory stats per module
        self.forward_memory: Dict[str, List[float]] = defaultdict(list)
        self.backward_memory: Dict[str, List[float]] = defaultdict(list)
        self.activation_sizes: Dict[str, List[float]] = defaultdict(list)
        
        # Store timing stats per module
        self.forward_time: Dict[str, List[float]] = defaultdict(list)
        self.backward_time: Dict[str, List[float]] = defaultdict(list)
        
        self.hooks = []
        self._register_hooks()
    
    def _get_memory_mb(self) -> float:
        """Get current allocated memory in MB"""
        if not self.is_cuda:
            return 0.0
        return torch.cuda.memory_allocated(self.device) / 2**20
    
    def _sync_time(self) -> float:
        """Get current time with GPU sync"""
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
        return time.perf_counter()
    
    def _register_hooks(self):
        """Register forward and backward hooks for all modules"""
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) > 0:
                # Skip container modules
                continue
            
            # Forward pre-hook to record start time
            def forward_pre_hook(module, input, name=name):
                module._fwd_start_time = self._sync_time()
            
            # Forward hook
            def forward_hook(module, input, output, name=name):
                # Record time
                if hasattr(module, '_fwd_start_time'):
                    elapsed = self._sync_time() - module._fwd_start_time
                    self.forward_time[name].append(elapsed * 1000)  # Convert to ms
                
                # Record memory
                mem = self._get_memory_mb()
                self.forward_memory[name].append(mem)
                
                # Track activation size
                if isinstance(output, torch.Tensor):
                    size_mb = output.numel() * output.element_size() / 2**20
                    self.activation_sizes[name].append(size_mb)
                elif isinstance(output, (tuple, list)) and len(output) > 0:
                    if isinstance(output[0], torch.Tensor):
                        size_mb = output[0].numel() * output[0].element_size() / 2**20
                        self.activation_sizes[name].append(size_mb)
            
            # Backward pre-hook to record start time
            def backward_pre_hook(module, grad_output, name=name):
                module._bwd_start_time = self._sync_time()
            
            # Backward hook
            def backward_hook(module, grad_input, grad_output, name=name):
                # Record time
                if hasattr(module, '_bwd_start_time'):
                    elapsed = self._sync_time() - module._bwd_start_time
                    self.backward_time[name].append(elapsed * 1000)  # Convert to ms
                
                # Record memory
                mem = self._get_memory_mb()
                self.backward_memory[name].append(mem)
            
            h0 = module.register_forward_pre_hook(forward_pre_hook)
            h1 = module.register_forward_hook(forward_hook)
            h2 = module.register_full_backward_pre_hook(backward_pre_hook)
            h3 = module.register_full_backward_hook(backward_hook)
            self.hooks.extend([h0, h1, h2, h3])
    
    def clear_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def reset_stats(self):
        """Reset all statistics"""
        self.forward_memory.clear()
        self.backward_memory.clear()
        self.activation_sizes.clear()
        self.forward_time.clear()
        self.backward_time.clear()
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics per module"""
        summary = {}
        
        all_modules = set(list(self.forward_memory.keys()) + 
                         list(self.backward_memory.keys()))
        
        for name in all_modules:
            fwd_mem = self.forward_memory.get(name, [])
            bwd_mem = self.backward_memory.get(name, [])
            act_size = self.activation_sizes.get(name, [])
            fwd_time = self.forward_time.get(name, [])
            bwd_time = self.backward_time.get(name, [])
            
            summary[name] = {
                'forward_memory_mb': sum(fwd_mem) / len(fwd_mem) if fwd_mem else 0,
                'backward_memory_mb': sum(bwd_mem) / len(bwd_mem) if bwd_mem else 0,
                'activation_size_mb': sum(act_size) / len(act_size) if act_size else 0,
                'forward_time_ms': sum(fwd_time) / len(fwd_time) if fwd_time else 0,
                'backward_time_ms': sum(bwd_time) / len(bwd_time) if bwd_time else 0,
                'total_memory_mb': (sum(fwd_mem) / len(fwd_mem) if fwd_mem else 0) + 
                                  (sum(bwd_mem) / len(bwd_mem) if bwd_mem else 0),
                'total_time_ms': (sum(fwd_time) / len(fwd_time) if fwd_time else 0) +
                                (sum(bwd_time) / len(bwd_time) if bwd_time else 0),
            }
        
        return summary
    
    def print_report(self, top_n: int = 20):
        """Print detailed memory report"""
        summary = self.get_summary()
        
        # Sort by total memory
        sorted_modules = sorted(summary.items(), 
                               key=lambda x: x[1]['total_memory_mb'], 
                               reverse=True)
        
        print("\n" + "="*120)
        print("DETAILED MEMORY & TIME PROFILING REPORT")
        print("="*120)
        print(f"{'Module Name':<45} {'Fwd(MB)':<12} {'Bwd(MB)':<12} {'Act(MB)':<12} {'Fwd(ms)':<12} {'Bwd(ms)':<12} {'Total(ms)':<12}")
        print("-"*120)
        
        for name, stats in sorted_modules[:top_n]:
            print(f"{name:<45} {stats['forward_memory_mb']:<12.2f} "
                  f"{stats['backward_memory_mb']:<12.2f} "
                  f"{stats['activation_size_mb']:<12.2f} "
                  f"{stats['forward_time_ms']:<12.3f} "
                  f"{stats['backward_time_ms']:<12.3f} "
                  f"{stats['total_time_ms']:<12.3f}")
        
        print("="*120)
        
        # Aggregate by module type
        self._print_aggregated_report(summary)
    
    def _print_aggregated_report(self, summary: Dict):
        """Print aggregated report by module type"""
        aggregated = defaultdict(lambda: {
            'forward': 0, 'backward': 0, 'activation': 0, 
            'forward_time': 0, 'backward_time': 0, 'count': 0
        })
        
        for name, stats in summary.items():
            # Extract module type
            if 'self_attn' in name:
                module_type = 'Attention'
            elif 'ffn' in name or 'fc' in name:
                module_type = 'FFN'
            elif 'lm_head' in name:
                module_type = 'LM_Head'
            elif 'embedding' in name:
                module_type = 'Embedding'
            elif 'norm' in name or 'ln_' in name:
                module_type = 'LayerNorm'
            else:
                module_type = 'Other'
            
            aggregated[module_type]['forward'] += stats['forward_memory_mb']
            aggregated[module_type]['backward'] += stats['backward_memory_mb']
            aggregated[module_type]['activation'] += stats['activation_size_mb']
            aggregated[module_type]['forward_time'] += stats['forward_time_ms']
            aggregated[module_type]['backward_time'] += stats['backward_time_ms']
            aggregated[module_type]['count'] += 1
        
        print("\n" + "="*120)
        print("AGGREGATED MEMORY & TIME BY MODULE TYPE")
        print("="*120)
        print(f"{'Type':<15} {'Count':<8} {'Fwd(MB)':<12} {'Bwd(MB)':<12} {'Act(MB)':<12} {'Fwd(ms)':<12} {'Bwd(ms)':<12} {'Total(ms)':<12}")
        print("-"*120)
        
        sorted_agg = sorted(aggregated.items(), 
                           key=lambda x: x[1]['forward'] + x[1]['backward'], 
                           reverse=True)
        
        for module_type, stats in sorted_agg:
            total_mem = stats['forward'] + stats['backward']
            total_time = stats['forward_time'] + stats['backward_time']
            print(f"{module_type:<15} {stats['count']:<8} {stats['forward']:<12.2f} "
                  f"{stats['backward']:<12.2f} {stats['activation']:<12.2f} "
                  f"{stats['forward_time']:<12.3f} {stats['backward_time']:<12.3f} "
                  f"{total_time:<12.3f}")
        
        print("="*120 + "\n")
        
        return aggregated


class ProfilingResults:
    """Store and visualize profiling results"""
    
    def __init__(self):
        self.results = []
    
    def add_result(self, name: str, config: Dict, profiler: DetailedMemoryProfiler, 
                   peak_memory: float, total_time: float):
        """Add profiling result"""
        summary = profiler.get_summary()
        aggregated = defaultdict(lambda: {
            'forward': 0, 'backward': 0, 'activation': 0, 
            'forward_time': 0, 'backward_time': 0, 'count': 0
        })
        
        for module_name, stats in summary.items():
            if 'self_attn' in module_name:
                module_type = 'Attention'
            elif 'ffn' in module_name or 'fc' in module_name:
                module_type = 'FFN'
            elif 'lm_head' in module_name:
                module_type = 'LM_Head'
            elif 'embedding' in module_name:
                module_type = 'Embedding'
            elif 'norm' in module_name or 'ln_' in module_name:
                module_type = 'LayerNorm'
            else:
                module_type = 'Other'
            
            aggregated[module_type]['forward'] += stats['forward_memory_mb']
            aggregated[module_type]['backward'] += stats['backward_memory_mb']
            aggregated[module_type]['activation'] += stats['activation_size_mb']
            aggregated[module_type]['forward_time'] += stats['forward_time_ms']
            aggregated[module_type]['backward_time'] += stats['backward_time_ms']
            aggregated[module_type]['count'] += 1
        
        self.results.append({
            'name': name,
            'config': config,
            'aggregated': dict(aggregated),
            'peak_memory': peak_memory,
            'total_time': total_time
        })
    
    def visualize(self, output_dir: str = "profiling_results"):
        """Create visualization comparing all results"""
        if not self.results:
            print("No results to visualize")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Extract data for plotting
        configs = []
        for result in self.results:
            config_str = f"{result['name']}\n(B={result['config']['batch_size']}, L={result['config']['seq_len']})"
            configs.append(config_str)
        
        # 1. Memory comparison by module type
        self._plot_memory_comparison(configs, output_path)
        
        # 2. Time comparison by module type
        self._plot_time_comparison(configs, output_path)
        
        # 3. Peak memory comparison
        self._plot_peak_memory(configs, output_path)
        
        # 4. Total time comparison
        self._plot_total_time(configs, output_path)
        
        # 5. Combined dashboard
        self._plot_dashboard(configs, output_path)
        
        print(f"\n✅ Visualizations saved to {output_path}/")
    
    def _plot_memory_comparison(self, configs, output_path):
        """Plot memory usage by module type"""
        module_types = ['Attention', 'FFN', 'LM_Head', 'Embedding', 'LayerNorm', 'Other']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        x = np.arange(len(configs))
        width = 0.15
        
        for i, mod_type in enumerate(module_types):
            forward_mem = [r['aggregated'].get(mod_type, {}).get('forward', 0) for r in self.results]
            backward_mem = [r['aggregated'].get(mod_type, {}).get('backward', 0) for r in self.results]
            
            ax1.bar(x + i*width, forward_mem, width, label=mod_type)
            ax2.bar(x + i*width, backward_mem, width, label=mod_type)
        
        ax1.set_title('Forward Memory by Module Type', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Memory (MB)', fontsize=12)
        ax1.set_xticks(x + width * 2.5)
        ax1.set_xticklabels(configs, fontsize=9)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        ax2.set_title('Backward Memory by Module Type', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Memory (MB)', fontsize=12)
        ax2.set_xticks(x + width * 2.5)
        ax2.set_xticklabels(configs, fontsize=9)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'memory_by_module.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_time_comparison(self, configs, output_path):
        """Plot time by module type"""
        module_types = ['Attention', 'FFN', 'LM_Head', 'Embedding', 'LayerNorm', 'Other']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        x = np.arange(len(configs))
        width = 0.15
        
        for i, mod_type in enumerate(module_types):
            forward_time = [r['aggregated'].get(mod_type, {}).get('forward_time', 0) for r in self.results]
            backward_time = [r['aggregated'].get(mod_type, {}).get('backward_time', 0) for r in self.results]
            
            ax1.bar(x + i*width, forward_time, width, label=mod_type)
            ax2.bar(x + i*width, backward_time, width, label=mod_type)
        
        ax1.set_title('Forward Time by Module Type', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Time (ms)', fontsize=12)
        ax1.set_xticks(x + width * 2.5)
        ax1.set_xticklabels(configs, fontsize=9)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        ax2.set_title('Backward Time by Module Type', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Time (ms)', fontsize=12)
        ax2.set_xticks(x + width * 2.5)
        ax2.set_xticklabels(configs, fontsize=9)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'time_by_module.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_peak_memory(self, configs, output_path):
        """Plot peak memory comparison"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        peak_mem = [r['peak_memory'] for r in self.results]
        colors = plt.cm.viridis(np.linspace(0, 1, len(configs)))
        
        bars = ax.bar(configs, peak_mem, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_title('Peak Memory Comparison', fontsize=16, fontweight='bold')
        ax.set_ylabel('Peak Memory (MB)', fontsize=12)
        ax.set_xlabel('Configuration', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=0, ha='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(output_path / 'peak_memory_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_total_time(self, configs, output_path):
        """Plot total time comparison"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        total_time = [r['total_time'] for r in self.results]
        colors = plt.cm.plasma(np.linspace(0, 1, len(configs)))
        
        bars = ax.bar(configs, total_time, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_title('Total Time Comparison', fontsize=16, fontweight='bold')
        ax.set_ylabel('Total Time (ms)', fontsize=12)
        ax.set_xlabel('Configuration', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=0, ha='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(output_path / 'total_time_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_dashboard(self, configs, output_path):
        """Create comprehensive dashboard"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Peak memory
        ax1 = fig.add_subplot(gs[0, :2])
        peak_mem = [r['peak_memory'] for r in self.results]
        ax1.bar(configs, peak_mem, color='steelblue', edgecolor='black')
        ax1.set_title('Peak Memory', fontsize=12, fontweight='bold')
        ax1.set_ylabel('MB')
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Total time
        ax2 = fig.add_subplot(gs[0, 2])
        total_time = [r['total_time'] for r in self.results]
        ax2.bar(configs, total_time, color='coral', edgecolor='black')
        ax2.set_title('Total Time', fontsize=12, fontweight='bold')
        ax2.set_ylabel('ms')
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Memory breakdown
        ax3 = fig.add_subplot(gs[1, :])
        module_types = ['Attention', 'FFN', 'LM_Head', 'Embedding', 'LayerNorm']
        x = np.arange(len(configs))
        width = 0.17
        
        for i, mod_type in enumerate(module_types):
            total_mem = [r['aggregated'].get(mod_type, {}).get('forward', 0) + 
                        r['aggregated'].get(mod_type, {}).get('backward', 0) 
                        for r in self.results]
            ax3.bar(x + i*width, total_mem, width, label=mod_type)
        
        ax3.set_title('Memory Breakdown by Module Type', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Memory (MB)')
        ax3.set_xticks(x + width * 2)
        ax3.set_xticklabels(configs, fontsize=9)
        ax3.legend(loc='upper left')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Time breakdown
        ax4 = fig.add_subplot(gs[2, :])
        for i, mod_type in enumerate(module_types):
            total_time = [r['aggregated'].get(mod_type, {}).get('forward_time', 0) + 
                         r['aggregated'].get(mod_type, {}).get('backward_time', 0) 
                         for r in self.results]
            ax4.bar(x + i*width, total_time, width, label=mod_type)
        
        ax4.set_title('Time Breakdown by Module Type', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Time (ms)')
        ax4.set_xticks(x + width * 2)
        ax4.set_xticklabels(configs, fontsize=9)
        ax4.legend(loc='upper left')
        ax4.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Comprehensive Profiling Dashboard', fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(output_path / 'dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()


def profile_model_memory(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: str = "cuda",
    num_iterations: int = 3,
    warmup_iter: int = 2
):
    """Profile model memory usage over multiple iterations"""
    
    print(f"\nProfiling model with:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Device: {device}")
    print(f"  Iterations: {num_iterations}")
    
    model = model.to(device)
    profiler = DetailedMemoryProfiler(model, device)

    # Warmup iterations
    if warmup_iter > 0:
        print(f"\n🔥 Running {warmup_iter} warmup iterations...")
        for _ in range(warmup_iter):
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"]
            loss.backward()
            model.zero_grad()

            del loss, outputs, input_ids, labels
            if device == "cuda":
                torch.cuda.empty_cache()

    # Clear stats after warmup and get initial memory
    profiler.reset_stats()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        initial_mem = torch.cuda.memory_allocated(device) / 2**20
        print(f"\nInitial memory after warmup: {initial_mem:.2f} MB")
    
    # Run iterations
    total_time = 0
    for i in range(num_iterations):
        print(f"\nIteration {i+1}/{num_iterations}")
        
        # Create dummy data
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        if device == "cuda":
            torch.cuda.synchronize(device)
            mem_before = torch.cuda.memory_allocated(device) / 2**20
        
        start_time = time.perf_counter()
        
        # Forward pass
        if device == "cuda":
            torch.cuda.synchronize(device)
        fwd_start = time.perf_counter()
        
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        
        if device == "cuda":
            torch.cuda.synchronize(device)
        fwd_time = (time.perf_counter() - fwd_start) * 1000
        
        if device == "cuda":
            mem_after_forward = torch.cuda.memory_allocated(device) / 2**20
            print(f"  Forward time: {fwd_time:.3f} ms")
            print(f"  Memory after forward: {mem_after_forward:.2f} MB (delta: {mem_after_forward - mem_before:.2f} MB)")
        
        # Backward pass
        if device == "cuda":
            torch.cuda.synchronize(device)
        bwd_start = time.perf_counter()
        
        loss.backward()
        
        if device == "cuda":
            torch.cuda.synchronize(device)
        bwd_time = (time.perf_counter() - bwd_start) * 1000
        
        if device == "cuda":
            mem_after_backward = torch.cuda.memory_allocated(device) / 2**20
            print(f"  Backward time: {bwd_time:.3f} ms")
            print(f"  Memory after backward: {mem_after_backward:.2f} MB (delta: {mem_after_backward - mem_after_forward:.2f} MB)")
            print(f"  Peak memory: {torch.cuda.max_memory_allocated(device) / 2**20:.2f} MB")
        
        iter_time = fwd_time + bwd_time
        total_time += iter_time
        print(f"  Iteration time: {iter_time:.3f} ms")
        
        # Clear gradients
        model.zero_grad()
        del loss, outputs, input_ids, labels
        
        if device == "cuda":
            torch.cuda.empty_cache()
    
    avg_time = total_time / num_iterations
    print(f"\n⏱️  Average iteration time: {avg_time:.3f} ms")
    
    # Print report
    profiler.print_report(top_n=30)
    
    # Get peak memory
    peak_memory = torch.cuda.max_memory_allocated(device) / 2**20 if device == "cuda" else 0
    
    return profiler, peak_memory, avg_time


def profile_with_flash_clipping(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    max_grad_norm: float = 1.0,
    device: str = "cuda",
    num_iterations: int = 3,
    use_triton: bool = True,
    warmup_iter: int = 2
):
    """Profile model with Flash/Ghost Clipping enabled"""
    
    print(f"\n{'='*100}")
    print(f"PROFILING WITH {'FLASH' if use_triton else 'GHOST'} CLIPPING")
    print(f"{'='*100}")
    
    model = model.to(device)
    
    # Wrap with GradSampleModule
    model = GradSampleModuleFastGradientClipping(model, use_triton=use_triton)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer = DPOptimizerFastGradientClipping(
        optimizer=optimizer,
        noise_multiplier=0.0,  # No noise for profiling
        max_grad_norm=max_grad_norm,
        expected_batch_size=batch_size,
    )
    
    profiler = DetailedMemoryProfiler(model, device)

    # Warmup iterations
    if warmup_iter > 0:
        print(f"\n🔥 Running {warmup_iter} warmup iterations...")
        for _ in range(warmup_iter):
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            del loss, outputs, input_ids, labels
            if device == "cuda":
                torch.cuda.empty_cache()

    # Clear stats after warmup and get initial memory
    profiler.reset_stats()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        initial_mem = torch.cuda.memory_allocated(device) / 2**20
        print(f"\nInitial memory after warmup: {initial_mem:.2f} MB")
    
    # Run iterations
    total_time = 0
    for i in range(num_iterations):
        print(f"\nIteration {i+1}/{num_iterations}")
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        if device == "cuda":
            torch.cuda.synchronize(device)
            mem_before = torch.cuda.memory_allocated(device) / 2**20
        
        # Forward
        if device == "cuda":
            torch.cuda.synchronize(device)
        fwd_start = time.perf_counter()
        
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        
        if device == "cuda":
            torch.cuda.synchronize(device)
        fwd_time = (time.perf_counter() - fwd_start) * 1000
        
        if device == "cuda":
            mem_after_forward = torch.cuda.memory_allocated(device) / 2**20
            print(f"  Forward time: {fwd_time:.3f} ms")
            print(f"  Memory after forward: {mem_after_forward:.2f} MB (delta: {mem_after_forward - mem_before:.2f} MB)")
        
        # Backward
        if device == "cuda":
            torch.cuda.synchronize(device)
        bwd_start = time.perf_counter()
        
        loss.backward()
        
        if device == "cuda":
            torch.cuda.synchronize(device)
        bwd_time = (time.perf_counter() - bwd_start) * 1000
        
        if device == "cuda":
            mem_after_backward = torch.cuda.memory_allocated(device) / 2**20
            print(f"  Backward time: {bwd_time:.3f} ms")
            print(f"  Memory after backward: {mem_after_backward:.2f} MB (delta: {mem_after_backward - mem_after_forward:.2f} MB)")
        
        # Optimizer step
        if device == "cuda":
            torch.cuda.synchronize(device)
        opt_start = time.perf_counter()
        
        optimizer.step()
        optimizer.zero_grad()
        
        if device == "cuda":
            torch.cuda.synchronize(device)
        opt_time = (time.perf_counter() - opt_start) * 1000
        
        if device == "cuda":
            mem_after_step = torch.cuda.memory_allocated(device) / 2**20
            print(f"  Optimizer time: {opt_time:.3f} ms")
            print(f"  Memory after optimizer step: {mem_after_step:.2f} MB")
            print(f"  Peak memory: {torch.cuda.max_memory_allocated(device) / 2**20:.2f} MB")
        
        iter_time = fwd_time + bwd_time + opt_time
        total_time += iter_time
        print(f"  Iteration time: {iter_time:.3f} ms")
        
        del loss, outputs, input_ids, labels
        if device == "cuda":
            torch.cuda.empty_cache()
    
    avg_time = total_time / num_iterations
    print(f"\n⏱️  Average iteration time: {avg_time:.3f} ms")
    
    # Print report
    profiler.print_report(top_n=30)
    
    # Get peak memory
    peak_memory = torch.cuda.max_memory_allocated(device) / 2**20 if device == "cuda" else 0
    
    return profiler, peak_memory, avg_time


def main():
    """Run comprehensive memory profiling"""
    
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    configs = [
        # large input length
        # {"vocab_size": 128, "hidden_dim": 128, "num_layers": 1, 
        #  "num_heads": 1, "seq_len": 4096*8, "batch_size": 2},

        # normal structure
        {"vocab_size": 32000, "hidden_dim": 768, "num_layers": 4,
        "num_heads": 12, "seq_len": 8192, "batch_size": 2},

        # large width
        # {"vocab_size": 32000, "hidden_dim": 4086, "num_layers": 1,
        # "num_heads": 1, "seq_len": 8192, "batch_size": 2},
    ]
    
    num_iter = 20
    warmup_iter = 10

    # Store results for visualization
    viz_results = ProfilingResults()
    
    for i, config in enumerate(configs):
        print(f"\n\n{'#'*100}")
        print(f"CONFIG {i+1}: {config}")
        print(f"{'#'*100}\n")
        
        # Create model
        model = SimpleBigModel(
            vocab_size=config["vocab_size"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            seq_len=config["seq_len"]
        )
        
        print(f"Model parameters: {model.count_parameters():,}")
        
        # Profile vanilla model
        print("\n" + "="*100)
        print("PROFILING VANILLA MODEL (NO DP)")
        print("="*100)
        profiler, peak_mem, avg_time = profile_model_memory(
            model=model,
            batch_size=config["batch_size"],
            seq_len=config["seq_len"],
            vocab_size=config["vocab_size"],
            device=device,
            num_iterations=num_iter,
            warmup_iter=warmup_iter
        )
        viz_results.add_result("Vanilla", config, profiler, peak_mem, avg_time)
        profiler.clear_hooks()
        
        # Profile with Ghost Clipping
        model = SimpleBigModel(
            vocab_size=config["vocab_size"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            seq_len=config["seq_len"]
        )
        profiler, peak_mem, avg_time = profile_with_flash_clipping(
            model=model,
            batch_size=config["batch_size"],
            seq_len=config["seq_len"],
            vocab_size=config["vocab_size"],
            device=device,
            num_iterations=num_iter,
            use_triton=False,  # Ghost clipping
            warmup_iter=warmup_iter
        )
        viz_results.add_result("Ghost Clipping", config, profiler, peak_mem, avg_time)
        profiler.clear_hooks()
        
        # Profile with Flash Clipping
        if device == "cuda":
            model = SimpleBigModel(
                vocab_size=config["vocab_size"],
                hidden_dim=config["hidden_dim"],
                num_layers=config["num_layers"],
                num_heads=config["num_heads"],
                seq_len=config["seq_len"]
            )
            profiler, peak_mem, avg_time = profile_with_flash_clipping(
                model=model,
                batch_size=config["batch_size"],
                seq_len=config["seq_len"],
                vocab_size=config["vocab_size"],
                device=device,
                num_iterations=num_iter,
                use_triton=True,  # Flash clipping
                warmup_iter=warmup_iter
            )
            viz_results.add_result("Flash Clipping", config, profiler, peak_mem, avg_time)
            profiler.clear_hooks()
        
        # Cleanup
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    
    # Generate visualizations
    print("\n" + "="*100)
    print("GENERATING VISUALIZATIONS")
    print("="*100)
    viz_results.visualize(output_dir="profiling_results")


if __name__ == "__main__":
    main()