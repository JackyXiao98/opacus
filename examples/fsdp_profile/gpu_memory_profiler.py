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
from typing import Dict, List, Optional, Tuple

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


@dataclass
class MemorySnapshot:
    """Memory snapshot at a specific point in time"""
    timestamp: float
    allocated: int
    reserved: int
    max_allocated: int
    max_reserved: int
    stage: str
    module: Optional[str] = None


@dataclass
class MemoryProfileResult:
    """Results from memory profiling"""
    seq_length: int
    batch_size: int
    max_physical_batch_size: int
    is_lora: bool
    lora_rank: int
    snapshots: List[MemorySnapshot]
    peak_memory: int
    memory_breakdown: Dict[str, int]
    execution_time: float


class GPUMemoryProfiler:
    """GPU Memory profiler for tracking memory usage across different stages"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.snapshots: List[MemorySnapshot] = []
        self.start_time = time.time()
        
    def reset(self):
        """Reset profiler state"""
        self.snapshots.clear()
        self.start_time = time.time()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        
    def snapshot(self, stage: str, module: Optional[str] = None):
        """Take a memory snapshot"""
        snapshot = MemorySnapshot(
            timestamp=time.time() - self.start_time,
            allocated=torch.cuda.memory_allocated(self.device),
            reserved=torch.cuda.memory_reserved(self.device),
            max_allocated=torch.cuda.max_memory_allocated(self.device),
            max_reserved=torch.cuda.max_memory_reserved(self.device),
            stage=stage,
            module=module
        )
        self.snapshots.append(snapshot)
        return snapshot
        
    @contextmanager
    def profile_stage(self, stage: str, module: Optional[str] = None):
        """Context manager for profiling a specific stage"""
        self.snapshot(f"{stage}_start", module)
        try:
            yield
        finally:
            self.snapshot(f"{stage}_end", module)
            
    def get_memory_breakdown(self) -> Dict[str, int]:
        """Calculate memory breakdown by stage"""
        breakdown = defaultdict(int)
        
        for i in range(len(self.snapshots) - 1):
            current = self.snapshots[i]
            next_snap = self.snapshots[i + 1]
            
            if current.stage.endswith('_start') and next_snap.stage.endswith('_end'):
                stage_name = current.stage.replace('_start', '')
                memory_diff = next_snap.max_allocated - current.allocated
                breakdown[stage_name] = max(breakdown[stage_name], memory_diff)
                
        return dict(breakdown)


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


def prepare_model(
    token: str,
    is_lora: bool = False,
    lora_rank: int = 16,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    profiler: GPUMemoryProfiler = None,
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

    # Count trainable parameters
    trainable_parameters = 0
    target_model = model_with_lora if is_lora else pretrained_model
    
    for name, param in target_model.named_parameters():
        if name == ("model.embed_tokens.weight"):
            param.requires_grad = False
        if param.requires_grad:
            trainable_parameters += param.numel()

    print(f"Trainable parameters: {trainable_parameters}")
    
    if profiler:
        profiler.snapshot("model_loading_end")
        
    return target_model, tokenizer


def profile_train_step(model, optimizer, criterion, batch, device, profiler: GPUMemoryProfiler):
    """Training step with detailed memory profiling"""
    
    with profiler.profile_stage("optimizer_zero_grad"):
        optimizer.zero_grad()
    
    with profiler.profile_stage("data_transfer"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
    
    with profiler.profile_stage("forward_pass"):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    with profiler.profile_stage("loss_computation"):
        loss = criterion(outputs.logits, labels)
    
    with profiler.profile_stage("backward_pass"):
        loss.backward()
    
    with profiler.profile_stage("optimizer_step"):
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    
    return loss


def run_memory_profile(
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
    num_steps: int = 5,
) -> MemoryProfileResult:
    """Run memory profiling for given parameters"""
    
    profiler = GPUMemoryProfiler(device)
    profiler.reset()
    
    start_time = time.time()
    
    # Model preparation
    model_final, tokenizer = prepare_model(
        token, is_lora, lora_rank, model_name, profiler
    )
    
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
    
    # Training steps
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
                    loss = profile_train_step(model, optimizer, criterion, batch, device, profiler)
                    
                print(f"Step {step}, Loss: {loss.item():.4f}, "
                      f"Memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
    
    execution_time = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated(device)
    memory_breakdown = profiler.get_memory_breakdown()
    
    return MemoryProfileResult(
        seq_length=seq_length,
        batch_size=batch_size,
        max_physical_batch_size=max_physical_batch_size,
        is_lora=is_lora,
        lora_rank=lora_rank,
        snapshots=profiler.snapshots,
        peak_memory=peak_memory,
        memory_breakdown=memory_breakdown,
        execution_time=execution_time,
    )


def launch_profile(
    rank: int,
    world_size: int,
    token: str,
    config_list: List[Dict],
    results_dir: str,
):
    """Launch profiling for multiple configurations"""
    torch.cuda.set_device(rank)
    setup(rank, world_size)
    
    master_process = rank == 0
    torch.manual_seed(1337 + rank)
    
    all_results = []
    
    for i, config in enumerate(config_list):
        if master_process:
            print(f"\n=== Running configuration {i+1}/{len(config_list)} ===")
            print(f"Config: {config}")
        
        try:
            result = run_memory_profile(
                token=token,
                rank=rank,
                world_size=world_size,
                device=torch.device(f"cuda:{rank}"),
                **config
            )
            all_results.append(result)
            
            if master_process:
                print(f"Peak memory: {result.peak_memory / 1024**3:.2f} GB")
                print(f"Execution time: {result.execution_time:.2f}s")
                
        except Exception as e:
            if master_process:
                print(f"Error in configuration {i+1}: {e}")
            continue
    
    # Save results
    if master_process and all_results:
        save_results(all_results, results_dir)
        create_visualizations(all_results, results_dir)
    
    cleanup()


def save_results(results: List[MemoryProfileResult], results_dir: str):
    """Save profiling results to files"""
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed results as JSON
    results_data = []
    for result in results:
        result_dict = {
            'seq_length': result.seq_length,
            'batch_size': result.batch_size,
            'max_physical_batch_size': result.max_physical_batch_size,
            'is_lora': result.is_lora,
            'lora_rank': result.lora_rank,
            'peak_memory_gb': result.peak_memory / 1024**3,
            'execution_time': result.execution_time,
            'memory_breakdown': {k: v / 1024**3 for k, v in result.memory_breakdown.items()},
            'snapshots': [
                {
                    'timestamp': s.timestamp,
                    'allocated_gb': s.allocated / 1024**3,
                    'reserved_gb': s.reserved / 1024**3,
                    'stage': s.stage,
                    'module': s.module
                }
                for s in result.snapshots
            ]
        }
        results_data.append(result_dict)
    
    with open(os.path.join(results_dir, 'memory_profile_results.json'), 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Save summary as CSV
    summary_data = []
    for result in results:
        summary_data.append({
            'seq_length': result.seq_length,
            'batch_size': result.batch_size,
            'max_physical_batch_size': result.max_physical_batch_size,
            'is_lora': result.is_lora,
            'lora_rank': result.lora_rank,
            'peak_memory_gb': result.peak_memory / 1024**3,
            'execution_time': result.execution_time,
            **{f'{k}_memory_gb': v / 1024**3 for k, v in result.memory_breakdown.items()}
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(os.path.join(results_dir, 'memory_profile_summary.csv'), index=False)
    print(f"Results saved to {results_dir}")


def create_visualizations(results: List[MemoryProfileResult], results_dir: str):
    """Create visualizations for memory profiling results"""
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Peak memory vs parameters
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Peak memory vs sequence length
    seq_lengths = [r.seq_length for r in results]
    peak_memories = [r.peak_memory / 1024**3 for r in results]
    
    axes[0, 0].scatter(seq_lengths, peak_memories, alpha=0.7, s=60)
    axes[0, 0].set_xlabel('Sequence Length')
    axes[0, 0].set_ylabel('Peak Memory (GB)')
    axes[0, 0].set_title('Peak Memory vs Sequence Length')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Peak memory vs batch size
    batch_sizes = [r.batch_size for r in results]
    axes[0, 1].scatter(batch_sizes, peak_memories, alpha=0.7, s=60)
    axes[0, 1].set_xlabel('Batch Size')
    axes[0, 1].set_ylabel('Peak Memory (GB)')
    axes[0, 1].set_title('Peak Memory vs Batch Size')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Memory breakdown by stage
    if results:
        stages = list(results[0].memory_breakdown.keys())
        stage_memories = {stage: [] for stage in stages}
        
        for result in results:
            for stage in stages:
                stage_memories[stage].append(result.memory_breakdown.get(stage, 0) / 1024**3)
        
        x_pos = np.arange(len(stages))
        width = 0.8 / len(results)
        
        for i, result in enumerate(results):
            memories = [result.memory_breakdown.get(stage, 0) / 1024**3 for stage in stages]
            axes[1, 0].bar(x_pos + i * width, memories, width, 
                          label=f'seq={result.seq_length}, bs={result.batch_size}', alpha=0.8)
        
        axes[1, 0].set_xlabel('Stage')
        axes[1, 0].set_ylabel('Memory Usage (GB)')
        axes[1, 0].set_title('Memory Usage by Stage')
        axes[1, 0].set_xticks(x_pos + width * (len(results) - 1) / 2)
        axes[1, 0].set_xticklabels(stages, rotation=45, ha='right')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Execution time vs memory
    exec_times = [r.execution_time for r in results]
    axes[1, 1].scatter(peak_memories, exec_times, alpha=0.7, s=60)
    axes[1, 1].set_xlabel('Peak Memory (GB)')
    axes[1, 1].set_ylabel('Execution Time (s)')
    axes[1, 1].set_title('Execution Time vs Peak Memory')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'memory_analysis_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Memory timeline for each configuration
    for i, result in enumerate(results):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        timestamps = [s.timestamp for s in result.snapshots]
        allocated = [s.allocated / 1024**3 for s in result.snapshots]
        reserved = [s.reserved / 1024**3 for s in result.snapshots]
        
        ax.plot(timestamps, allocated, label='Allocated Memory', linewidth=2)
        ax.plot(timestamps, reserved, label='Reserved Memory', linewidth=2, alpha=0.7)
        
        # Add stage annotations
        for snapshot in result.snapshots:
            if snapshot.stage.endswith('_start'):
                ax.axvline(x=snapshot.timestamp, color='red', linestyle='--', alpha=0.5)
                ax.text(snapshot.timestamp, max(allocated) * 0.9, 
                       snapshot.stage.replace('_start', ''), 
                       rotation=90, fontsize=8, ha='right')
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Memory Usage (GB)')
        ax.set_title(f'Memory Timeline - seq_len={result.seq_length}, batch_size={result.batch_size}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'memory_timeline_config_{i+1}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Heatmap of memory usage
    if len(results) > 1:
        # Create a pivot table for heatmap
        heatmap_data = []
        for result in results:
            heatmap_data.append({
                'seq_length': result.seq_length,
                'batch_size': result.batch_size,
                'peak_memory': result.peak_memory / 1024**3
            })
        
        df_heatmap = pd.DataFrame(heatmap_data)
        pivot_table = df_heatmap.pivot_table(
            values='peak_memory', 
            index='seq_length', 
            columns='batch_size', 
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Peak Memory (GB)'})
        plt.title('Peak Memory Usage Heatmap')
        plt.xlabel('Batch Size')
        plt.ylabel('Sequence Length')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'memory_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="GPU Memory Profiler for Opacus")
    parser.add_argument("--token", type=str, required=True, help="Huggingface token")
    parser.add_argument("--results_dir", type=str, default="./memory_profile_results", 
                       help="Directory to save results")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", 
                       help="Model name")
    parser.add_argument("--num_steps", type=int, default=5, 
                       help="Number of training steps per configuration")
    
    # Parameter ranges for testing
    parser.add_argument("--seq_lengths", nargs='+', type=int, default=[128, 256, 512], 
                       help="Sequence lengths to test")
    parser.add_argument("--batch_sizes", nargs='+', type=int, default=[8, 16, 32], 
                       help="Batch sizes to test")
    parser.add_argument("--max_physical_batch_sizes", nargs='+', type=int, default=[1, 2, 4], 
                       help="Max physical batch sizes to test")
    parser.add_argument("--lora_ranks", nargs='+', type=int, default=[8, 16], 
                       help="LoRA ranks to test")
    
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    print(f"CUDA available: {torch.cuda.is_available()}, devices: {world_size}")
    
    # Generate configuration combinations
    config_list = []
    for seq_len in args.seq_lengths:
        for batch_size in args.batch_sizes:
            for max_phys_batch in args.max_physical_batch_sizes:
                for lora_rank in args.lora_ranks:
                    # Test without LoRA by default, and with LoRA
                    for is_lora in [False, True]:
                        if max_phys_batch <= batch_size:  # Ensure valid configuration
                            config_list.append({
                                'seq_length': seq_len,
                                'batch_size': batch_size,
                                'max_physical_batch_size': max_phys_batch,
                                'is_lora': is_lora,
                                'lora_rank': lora_rank if is_lora else 0,
                                'model_name': args.model_name,
                                'num_steps': args.num_steps,
                            })
    
    print(f"Testing {len(config_list)} configurations")
    
    # Launch profiling
    mp.spawn(
        launch_profile,
        args=(world_size, args.token, config_list, args.results_dir),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()