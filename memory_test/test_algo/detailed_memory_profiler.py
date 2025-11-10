#!/usr/bin/env python3
"""
Detailed Memory Profiler for DP-SGD Components

This profiler tracks fine-grained memory usage:
1. Model parameters
2. Optimizer states
3. Forward activations (standard vs DP-SGD hooks)
4. Backward gradients
5. DP-SGD specific: norm_samples, ggT/aaT matrices, etc.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import json
import time
from collections import defaultdict
from pathlib import Path


class DetailedMemorySnapshot:
    """A snapshot of memory usage at a specific point in time"""
    
    def __init__(self, name: str, device: str = "cuda"):
        self.name = name
        self.device = device
        self.timestamp = time.time()
        
        if device == "cuda" and torch.cuda.is_available():
            self.allocated = torch.cuda.memory_allocated(device) / 2**20
            self.reserved = torch.cuda.memory_reserved(device) / 2**20
            self.max_allocated = torch.cuda.max_memory_allocated(device) / 2**20
        else:
            self.allocated = 0
            self.reserved = 0
            self.max_allocated = 0
    
    def to_dict(self):
        return {
            "name": self.name,
            "allocated_mb": self.allocated,
            "reserved_mb": self.reserved,
            "max_allocated_mb": self.max_allocated,
            "timestamp": self.timestamp
        }


class EnhancedMemoryProfiler:
    """
    Enhanced profiler that tracks DP-SGD specific memory components
    """
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.is_cuda = device == "cuda" and torch.cuda.is_available()
        
        # Memory snapshots at different stages
        self.snapshots: List[DetailedMemorySnapshot] = []
        
        # Track specific components
        self.component_memory: Dict[str, List[float]] = defaultdict(list)
        
        # Hooks
        self.hooks = []
        self.activation_memory = 0
        self.norm_sample_memory = 0
        self.temp_matrix_memory = 0
        
    def reset(self):
        """Reset all statistics"""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()
        self.snapshots.clear()
        self.component_memory.clear()
        self.activation_memory = 0
        self.norm_sample_memory = 0
        self.temp_matrix_memory = 0
    
    def take_snapshot(self, name: str):
        """Take a memory snapshot at current point"""
        snapshot = DetailedMemorySnapshot(name, self.device)
        self.snapshots.append(snapshot)
        return snapshot
    
    def register_component_hooks(self):
        """Register hooks to track DP-SGD component memory"""
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) > 0:
                continue
            
            # Forward hook to track activations
            def forward_hook(mod, inp, out, module_name=name):
                if hasattr(mod, 'activations') and len(mod.activations) > 0:
                    # This module has DP-SGD activation hooks
                    for act_list in mod.activations:
                        for act in act_list:
                            if isinstance(act, torch.Tensor):
                                size_mb = act.numel() * act.element_size() / 2**20
                                self.activation_memory += size_mb
            
            # Backward hook to track norm samples
            def backward_hook(mod, grad_in, grad_out, module_name=name):
                # Check for norm_sample attribute (DP-SGD specific)
                for param_name, param in mod.named_parameters():
                    if hasattr(param, '_norm_sample'):
                        size_mb = param._norm_sample.numel() * param._norm_sample.element_size() / 2**20
                        self.norm_sample_memory += size_mb
            
            h1 = module.register_forward_hook(forward_hook)
            h2 = module.register_full_backward_hook(backward_hook)
            self.hooks.extend([h1, h2])
    
    def clear_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def estimate_model_memory(self):
        """Estimate memory used by model parameters"""
        param_memory = 0
        for param in self.model.parameters():
            param_memory += param.numel() * param.element_size() / 2**20
        return param_memory
    
    def estimate_gradient_memory(self):
        """Estimate memory used by gradients"""
        grad_memory = 0
        for param in self.model.parameters():
            if param.grad is not None:
                grad_memory += param.grad.numel() * param.grad.element_size() / 2**20
        return grad_memory
    
    def estimate_optimizer_memory(self, optimizer):
        """Estimate memory used by optimizer states"""
        opt_memory = 0
        for state in optimizer.state.values():
            for v in state.values():
                if isinstance(v, torch.Tensor):
                    opt_memory += v.numel() * v.element_size() / 2**20
        return opt_memory
    
    def get_detailed_breakdown(self, optimizer=None):
        """Get detailed memory breakdown"""
        breakdown = {
            "model_parameters_mb": self.estimate_model_memory(),
            "gradients_mb": self.estimate_gradient_memory(),
            "activation_hooks_mb": self.activation_memory,
            "norm_samples_mb": self.norm_sample_memory,
            "temp_matrices_mb": self.temp_matrix_memory,
        }
        
        if optimizer is not None:
            breakdown["optimizer_states_mb"] = self.estimate_optimizer_memory(optimizer)
        
        if self.is_cuda:
            breakdown["total_allocated_mb"] = torch.cuda.memory_allocated(self.device) / 2**20
            breakdown["total_reserved_mb"] = torch.cuda.memory_reserved(self.device) / 2**20
            breakdown["peak_allocated_mb"] = torch.cuda.max_memory_allocated(self.device) / 2**20
        
        return breakdown
    
    def save_results(self, output_path: str, optimizer=None):
        """Save profiling results to JSON"""
        results = {
            "snapshots": [s.to_dict() for s in self.snapshots],
            "breakdown": self.get_detailed_breakdown(optimizer),
            "component_memory": dict(self.component_memory)
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved to {output_path}")
        return results


def print_memory_breakdown(breakdown: Dict):
    """Pretty print memory breakdown"""
    print("\n" + "="*80)
    print("DETAILED MEMORY BREAKDOWN")
    print("="*80)
    
    # Sort by size
    items = [(k, v) for k, v in breakdown.items() if k.endswith('_mb')]
    items.sort(key=lambda x: x[1], reverse=True)
    
    for name, value in items:
        percentage = 0
        if 'peak_allocated_mb' in breakdown and breakdown['peak_allocated_mb'] > 0:
            percentage = (value / breakdown['peak_allocated_mb']) * 100
        
        display_name = name.replace('_mb', '').replace('_', ' ').title()
        print(f"  {display_name:<30} {value:>10.2f} MB  ({percentage:>5.1f}%)")
    
    print("="*80 + "\n")

