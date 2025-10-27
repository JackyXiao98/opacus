#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import load_dataset
from huggingface_hub import login
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.utils.fsdp_utils import FSDP2Wrapper
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn as nn


@dataclass
class DetailedMemorySnapshot:
    """Detailed memory snapshot with module-level breakdown"""
    timestamp: float
    allocated: int
    reserved: int
    max_allocated: int
    max_reserved: int
    stage: str
    module: Optional[str] = None
    component: Optional[str] = None  # attention, mlp, embedding, etc.
    operation: Optional[str] = None  # forward, backward, optimizer
    tensor_info: Optional[Dict[str, Any]] = None


class ModuleMemoryTracker:
    """Track memory usage for specific modules and operations"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.module_snapshots = defaultdict(list)
        self.hooks = []
        self.current_stage = "unknown"
        
    def set_stage(self, stage: str):
        """Set current profiling stage"""
        self.current_stage = stage
        
    def register_hooks(self, model: nn.Module):
        """Register forward and backward hooks for detailed tracking"""
        
        def create_forward_hook(module_name: str, component: str):
            def forward_hook(module, input, output):
                self._record_memory(f"{module_name}_forward", component, "forward")
                return output
            return forward_hook
            
        def create_backward_hook(module_name: str, component: str):
            def backward_hook(module, grad_input, grad_output):
                self._record_memory(f"{module_name}_backward", component, "backward")
                return grad_input
            return backward_hook
        
        # Register hooks for different components
        for name, module in model.named_modules():
            component = self._classify_component(name, module)
            if component:
                # Forward hook
                hook = module.register_forward_hook(
                    create_forward_hook(name, component)
                )
                self.hooks.append(hook)
                
                # Backward hook
                hook = module.register_full_backward_hook(
                    create_backward_hook(name, component)
                )
                self.hooks.append(hook)
    
    def _classify_component(self, name: str, module: nn.Module) -> Optional[str]:
        """Classify module into component categories"""
        name_lower = name.lower()
        
        if 'attention' in name_lower or 'attn' in name_lower:
            return 'attention'
        elif 'mlp' in name_lower or 'feed_forward' in name_lower or 'ffn' in name_lower:
            return 'mlp'
        elif 'embed' in name_lower:
            return 'embedding'
        elif 'norm' in name_lower or 'layer_norm' in name_lower:
            return 'normalization'
        elif 'lora' in name_lower:
            return 'lora'
        elif isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            return 'linear'
        elif isinstance(module, (nn.Dropout,)):
            return 'dropout'
        else:
            return None
    
    def _record_memory(self, module_name: str, component: str, operation: str):
        """Record memory usage for a specific module operation"""
        snapshot = DetailedMemorySnapshot(
            timestamp=time.time(),
            allocated=torch.cuda.memory_allocated(self.device),
            reserved=torch.cuda.memory_reserved(self.device),
            max_allocated=torch.cuda.max_memory_allocated(self.device),
            max_reserved=torch.cuda.max_memory_reserved(self.device),
            stage=self.current_stage,
            module=module_name,
            component=component,
            operation=operation,
        )
        self.module_snapshots[component].append(snapshot)
    
    def cleanup_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def get_component_memory_breakdown(self) -> Dict[str, Dict[str, int]]:
        """Get memory breakdown by component and operation"""
        breakdown = defaultdict(lambda: defaultdict(int))
        
        for component, snapshots in self.module_snapshots.items():
            for i in range(len(snapshots) - 1):
                current = snapshots[i]
                next_snap = snapshots[i + 1]
                
                memory_diff = next_snap.allocated - current.allocated
                if memory_diff > 0:
                    breakdown[component][current.operation] += memory_diff
        
        return {k: dict(v) for k, v in breakdown.items()}


class AdvancedGPUMemoryProfiler:
    """Advanced GPU Memory profiler with detailed component analysis"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.snapshots: List[DetailedMemorySnapshot] = []
        self.module_tracker = ModuleMemoryTracker(device)
        self.start_time = time.time()
        self.gradient_memory = 0
        self.activation_memory = 0
        
    def reset(self):
        """Reset profiler state"""
        self.snapshots.clear()
        self.module_tracker.module_snapshots.clear()
        self.start_time = time.time()
        self.gradient_memory = 0
        self.activation_memory = 0
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        
    def register_model_hooks(self, model: nn.Module):
        """Register hooks for detailed model tracking"""
        self.module_tracker.register_hooks(model)
        
    def cleanup(self):
        """Cleanup hooks and resources"""
        self.module_tracker.cleanup_hooks()
        
    def snapshot(self, stage: str, module: Optional[str] = None, component: Optional[str] = None):
        """Take a detailed memory snapshot"""
        self.module_tracker.set_stage(stage)
        
        snapshot = DetailedMemorySnapshot(
            timestamp=time.time() - self.start_time,
            allocated=torch.cuda.memory_allocated(self.device),
            reserved=torch.cuda.memory_reserved(self.device),
            max_allocated=torch.cuda.max_memory_allocated(self.device),
            max_reserved=torch.cuda.max_memory_reserved(self.device),
            stage=stage,
            module=module,
            component=component,
        )
        self.snapshots.append(snapshot)
        return snapshot
        
    @contextmanager
    def profile_stage(self, stage: str, module: Optional[str] = None, component: Optional[str] = None):
        """Context manager for profiling a specific stage"""
        self.snapshot(f"{stage}_start", module, component)
        try:
            yield
        finally:
            self.snapshot(f"{stage}_end", module, component)
            
    def estimate_gradient_memory(self, model: nn.Module) -> int:
        """Estimate memory used by gradients"""
        total_params = 0
        for param in model.parameters():
            if param.requires_grad:
                total_params += param.numel()
        
        # Assume float32 gradients (4 bytes per parameter)
        return total_params * 4
    
    def estimate_activation_memory(self, batch_size: int, seq_length: int, hidden_size: int, num_layers: int) -> int:
        """Estimate memory used by activations"""
        # Rough estimation for transformer activations
        # This is a simplified calculation
        attention_memory = batch_size * seq_length * seq_length * num_layers * 4  # attention matrices
        hidden_memory = batch_size * seq_length * hidden_size * num_layers * 4  # hidden states
        return attention_memory + hidden_memory
        
    def get_detailed_memory_breakdown(self) -> Dict[str, Any]:
        """Get comprehensive memory breakdown"""
        breakdown = {
            'stages': defaultdict(int),
            'components': self.module_tracker.get_component_memory_breakdown(),
            'gradient_estimate': self.gradient_memory,
            'activation_estimate': self.activation_memory,
        }
        
        # Calculate stage-wise memory usage
        for i in range(len(self.snapshots) - 1):
            current = self.snapshots[i]
            next_snap = self.snapshots[i + 1]
            
            if current.stage.endswith('_start') and next_snap.stage.endswith('_end'):
                stage_name = current.stage.replace('_start', '')
                memory_diff = next_snap.max_allocated - current.allocated
                breakdown['stages'][stage_name] = max(breakdown['stages'][stage_name], memory_diff)
        
        breakdown['stages'] = dict(breakdown['stages'])
        return breakdown


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def prepare_snli_dataset(tokenizer, split="train", max_len=128):
    dataset = load_dataset("snli", split=split)
    dataset = dataset.filter(lambda example: example["label"] != -1)

    def tokenize_function(example):
        return tokenizer(
            example["premise"],
            example["hypothesis"],
            padding="max_length",
            max_length=max_len,
            truncation=True,
        )

    encoded_dataset = dataset.map(tokenize_function, batched=True)
    encoded_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    return encoded_dataset


def prepare_model_with_profiling(
    token: str,
    is_lora: bool = False,
    lora_rank: int = 16,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    profiler: AdvancedGPUMemoryProfiler = None,
):
    if profiler:
        profiler.snapshot("model_loading_start")
        
    login(token)
    
    with profiler.profile_stage("model_download") if profiler else contextmanager(lambda: iter([None]))():
        pretrained_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=3
        )
        
    with profiler.profile_stage("tokenizer_loading") if profiler else contextmanager(lambda: iter([None]))():
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            pretrained_model.config.pad_token_id = tokenizer.pad_token_id
    
    if is_lora:
        with profiler.profile_stage("lora_setup") if profiler else contextmanager(lambda: iter([None]))():
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_rank,
                lora_alpha=32,
                lora_dropout=0.05,
            )
            model_with_lora = get_peft_model(pretrained_model, lora_config)

    target_model = model_with_lora if is_lora else pretrained_model
    
    # Count trainable parameters and estimate gradient memory
    trainable_parameters = 0
    for name, param in target_model.named_parameters():
        if name == ("model.embed_tokens.weight"):
            param.requires_grad = False
        if param.requires_grad:
            trainable_parameters += param.numel()

    if profiler:
        profiler.gradient_memory = profiler.estimate_gradient_memory(target_model)
        profiler.snapshot("model_loading_end")
        
    print(f"Trainable parameters: {trainable_parameters}")
    print(f"Estimated gradient memory: {profiler.gradient_memory / 1024**3:.2f} GB" if profiler else "")
        
    return target_model, tokenizer


def detailed_train_step(model, optimizer, criterion, batch, device, profiler: AdvancedGPUMemoryProfiler):
    """Training step with very detailed memory profiling"""
    
    with profiler.profile_stage("optimizer_zero_grad"):
        optimizer.zero_grad()
    
    with profiler.profile_stage("data_transfer"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
    
    # Estimate activation memory
    batch_size, seq_length = input_ids.shape
    hidden_size = model.config.hidden_size if hasattr(model, 'config') else 4096
    num_layers = model.config.num_hidden_layers if hasattr(model, 'config') else 32
    profiler.activation_memory = profiler.estimate_activation_memory(
        batch_size, seq_length, hidden_size, num_layers
    )
    
    with profiler.profile_stage("forward_pass", component="full_model"):
        # Enable gradient computation for activation memory tracking
        with torch.enable_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    with profiler.profile_stage("loss_computation"):
        loss = criterion(outputs.logits, labels)
    
    with profiler.profile_stage("backward_pass", component="full_model"):
        loss.backward()
    
    with profiler.profile_stage("optimizer_step"):
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    
    # Force garbage collection to get accurate memory measurements
    with profiler.profile_stage("garbage_collection"):
        gc.collect()
        torch.cuda.empty_cache()
    
    return loss


def run_detailed_memory_profile(
    token: str,
    rank: int,
    world_size: int,
    device: torch.device,
    seq_length: int = 128,
    batch_size: int = 32,
    max_physical_batch_size: int = 1,
    is_lora: bool = False,
    lora_rank: int = 16,
    learning_rate: float = 1e-5,
    sigma: float = 1.0,
    max_grad_norm: float = 1.0,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    mp_policy: dist.fsdp.MixedPrecisionPolicy = None,
    num_steps: int = 3,
) -> Dict[str, Any]:
    """Run detailed memory profiling with component breakdown"""
    
    profiler = AdvancedGPUMemoryProfiler(device)
    profiler.reset()
    
    start_time = time.time()
    
    try:
        # Model preparation
        model_final, tokenizer = prepare_model_with_profiling(
            token, is_lora, lora_rank, model_name, profiler
        )
        
        # Register hooks for detailed tracking
        profiler.register_model_hooks(model_final)
        
        # Dataset preparation
        with profiler.profile_stage("dataset_preparation"):
            train_dataset = prepare_snli_dataset(tokenizer, split="train", max_len=seq_length)
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size // world_size,
                sampler=DistributedSampler(train_dataset),
            )
        
        # FSDP wrapping
        with profiler.profile_stage("fsdp_wrapping"):
            model = FSDP2Wrapper(model_final, mp_policy=mp_policy)
            model.train()
        
        # Optimizer setup
        with profiler.profile_stage("optimizer_setup"):
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        
        # Privacy engine setup
        with profiler.profile_stage("privacy_engine_setup"):
            privacy_engine = PrivacyEngine()
            model, optimizer, criterion, train_dataloader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_dataloader,
                noise_multiplier=sigma,
                max_grad_norm=max_grad_norm,
                grad_sample_mode="ghost_fsdp",
                criterion=torch.nn.CrossEntropyLoss(),
                poisson_sampling=False,
            )
        
        # Training steps with detailed profiling
        with profiler.profile_stage("training_loop"):
            with BatchMemoryManager(
                data_loader=train_dataloader,
                max_physical_batch_size=max_physical_batch_size,
                optimizer=optimizer,
            ) as memory_safe_data_loader:
                
                for step, batch in enumerate(memory_safe_data_loader):
                    if step >= num_steps:
                        break
                        
                    with profiler.profile_stage(f"training_step_{step}"):
                        loss = detailed_train_step(model, optimizer, criterion, batch, device, profiler)
                        
                    current_memory = torch.cuda.memory_allocated(device) / 1024**3
                    peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3
                    
                    print(f"Step {step}: Loss={loss.item():.4f}, "
                          f"Current={current_memory:.2f}GB, Peak={peak_memory:.2f}GB")
        
        execution_time = time.time() - start_time
        peak_memory = torch.cuda.max_memory_allocated(device)
        detailed_breakdown = profiler.get_detailed_memory_breakdown()
        
        result = {
            'config': {
                'seq_length': seq_length,
                'batch_size': batch_size,
                'max_physical_batch_size': max_physical_batch_size,
                'is_lora': is_lora,
                'lora_rank': lora_rank,
            },
            'peak_memory': peak_memory,
            'execution_time': execution_time,
            'detailed_breakdown': detailed_breakdown,
            'snapshots': profiler.snapshots,
        }
        
        return result
        
    finally:
        profiler.cleanup()


def create_detailed_visualizations(results: List[Dict[str, Any]], results_dir: str):
    """Create detailed visualizations for component-level analysis"""
    
    os.makedirs(results_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8')
    
    # 1. Component memory breakdown
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Collect component data
    all_components = set()
    for result in results:
        components = result['detailed_breakdown']['components']
        all_components.update(components.keys())
    
    component_data = defaultdict(list)
    config_labels = []
    
    for result in results:
        config = result['config']
        label = f"seq={config['seq_length']}, bs={config['batch_size']}"
        config_labels.append(label)
        
        components = result['detailed_breakdown']['components']
        for comp in all_components:
            total_memory = sum(components.get(comp, {}).values())
            component_data[comp].append(total_memory / 1024**3)
    
    # Component memory usage bar chart
    x_pos = np.arange(len(config_labels))
    width = 0.8 / len(all_components)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_components)))
    
    for i, (comp, memories) in enumerate(component_data.items()):
        axes[0, 0].bar(x_pos + i * width, memories, width, 
                      label=comp, color=colors[i], alpha=0.8)
    
    axes[0, 0].set_xlabel('Configuration')
    axes[0, 0].set_ylabel('Memory Usage (GB)')
    axes[0, 0].set_title('Memory Usage by Component')
    axes[0, 0].set_xticks(x_pos + width * (len(all_components) - 1) / 2)
    axes[0, 0].set_xticklabels(config_labels, rotation=45, ha='right')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Memory breakdown pie chart for first configuration
    if results:
        first_result = results[0]
        breakdown = first_result['detailed_breakdown']
        
        # Pie chart data
        pie_data = {}
        pie_data['Gradients'] = breakdown['gradient_estimate'] / 1024**3
        pie_data['Activations'] = breakdown['activation_estimate'] / 1024**3
        
        for comp, ops in breakdown['components'].items():
            pie_data[f'{comp.title()}'] = sum(ops.values()) / 1024**3
        
        # Filter out very small components
        pie_data = {k: v for k, v in pie_data.items() if v > 0.1}
        
        axes[0, 1].pie(pie_data.values(), labels=pie_data.keys(), autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title(f'Memory Distribution - {config_labels[0]}')
    
    # Peak memory vs sequence length
    seq_lengths = [r['config']['seq_length'] for r in results]
    peak_memories = [r['peak_memory'] / 1024**3 for r in results]
    
    axes[1, 0].scatter(seq_lengths, peak_memories, alpha=0.7, s=80, c='red')
    axes[1, 0].set_xlabel('Sequence Length')
    axes[1, 0].set_ylabel('Peak Memory (GB)')
    axes[1, 0].set_title('Peak Memory vs Sequence Length')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Memory efficiency (memory per parameter)
    batch_sizes = [r['config']['batch_size'] for r in results]
    axes[1, 1].scatter(batch_sizes, peak_memories, alpha=0.7, s=80, c='blue')
    axes[1, 1].set_xlabel('Batch Size')
    axes[1, 1].set_ylabel('Peak Memory (GB)')
    axes[1, 1].set_title('Peak Memory vs Batch Size')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'detailed_memory_analysis.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Operation-level breakdown
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Collect operation data
    operation_data = defaultdict(lambda: defaultdict(list))
    
    for result in results:
        config = result['config']
        label = f"seq={config['seq_length']}, bs={config['batch_size']}"
        
        components = result['detailed_breakdown']['components']
        for comp, ops in components.items():
            for op, memory in ops.items():
                operation_data[comp][op].append(memory / 1024**3)
    
    # Create stacked bar chart
    bottom = np.zeros(len(config_labels))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    color_idx = 0
    
    for comp, ops in operation_data.items():
        for op, memories in ops.items():
            if len(memories) == len(config_labels):
                ax.bar(config_labels, memories, bottom=bottom, 
                      label=f'{comp}_{op}', color=colors[color_idx % 20], alpha=0.8)
                bottom += np.array(memories)
                color_idx += 1
    
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Memory Usage (GB)')
    ax.set_title('Memory Usage by Component and Operation')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'operation_level_breakdown.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed visualizations saved to {results_dir}")


def launch_detailed_profile(
    rank: int,
    world_size: int,
    token: str,
    config_list: List[Dict],
    results_dir: str,
):
    """Launch detailed profiling"""
    torch.cuda.set_device(rank)
    setup(rank, world_size)
    
    master_process = rank == 0
    torch.manual_seed(1337 + rank)
    
    all_results = []
    
    for i, config in enumerate(config_list):
        if master_process:
            print(f"\n=== Detailed profiling {i+1}/{len(config_list)} ===")
            print(f"Config: {config}")
        
        try:
            result = run_detailed_memory_profile(
                token=token,
                rank=rank,
                world_size=world_size,
                device=torch.device(f"cuda:{rank}"),
                **config
            )
            all_results.append(result)
            
            if master_process:
                print(f"Peak memory: {result['peak_memory'] / 1024**3:.2f} GB")
                print(f"Execution time: {result['execution_time']:.2f}s")
                
                # Print component breakdown
                components = result['detailed_breakdown']['components']
                print("Component breakdown:")
                for comp, ops in components.items():
                    total = sum(ops.values()) / 1024**3
                    print(f"  {comp}: {total:.2f} GB")
                
        except Exception as e:
            if master_process:
                print(f"Error in configuration {i+1}: {e}")
            continue
    
    # Save detailed results
    if master_process and all_results:
        # Save as JSON
        with open(os.path.join(results_dir, 'detailed_memory_results.json'), 'w') as f:
            # Convert snapshots to serializable format
            serializable_results = []
            for result in all_results:
                serializable_result = result.copy()
                serializable_result['snapshots'] = [
                    {
                        'timestamp': s.timestamp,
                        'allocated_gb': s.allocated / 1024**3,
                        'reserved_gb': s.reserved / 1024**3,
                        'stage': s.stage,
                        'module': s.module,
                        'component': s.component,
                        'operation': s.operation,
                    }
                    for s in result['snapshots']
                ]
                serializable_results.append(serializable_result)
            
            json.dump(serializable_results, f, indent=2)
        
        create_detailed_visualizations(all_results, results_dir)
        print(f"Detailed results saved to {results_dir}")
    
    cleanup()


def main():
    parser = argparse.ArgumentParser(description="Detailed GPU Memory Profiler for Opacus")
    parser.add_argument("--token", type=str, required=True, help="Huggingface token")
    parser.add_argument("--results_dir", type=str, default="./detailed_memory_results", 
                       help="Directory to save results")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", 
                       help="Model name")
    parser.add_argument("--num_steps", type=int, default=3, 
                       help="Number of training steps per configuration")
    
    # Focused parameter ranges for detailed analysis
    parser.add_argument("--seq_lengths", nargs='+', type=int, default=[128, 256], 
                       help="Sequence lengths to test")
    parser.add_argument("--batch_sizes", nargs='+', type=int, default=[8, 16], 
                       help="Batch sizes to test")
    parser.add_argument("--max_physical_batch_sizes", nargs='+', type=int, default=[1, 2], 
                       help="Max physical batch sizes to test")
    
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    print(f"CUDA available: {torch.cuda.is_available()}, devices: {world_size}")
    
    # Generate focused configuration combinations
    config_list = []
    for seq_len in args.seq_lengths:
        for batch_size in args.batch_sizes:
            for max_phys_batch in args.max_physical_batch_sizes:
                if max_phys_batch <= batch_size:
                    # Test without LoRA by default for detailed analysis
                    config_list.append({
                        'seq_length': seq_len,
                        'batch_size': batch_size,
                        'max_physical_batch_size': max_phys_batch,
                        'is_lora': False,
                        'lora_rank': 16,
                        'model_name': args.model_name,
                        'num_steps': args.num_steps,
                    })
    
    print(f"Running detailed analysis on {len(config_list)} configurations")
    
    # Launch detailed profiling
    mp.spawn(
        launch_detailed_profile,
        args=(world_size, args.token, config_list, args.results_dir),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()