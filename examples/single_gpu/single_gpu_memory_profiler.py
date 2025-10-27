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
from datasets import load_dataset
from huggingface_hub import login
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
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
    """Result of memory profiling for a specific configuration"""
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
    """GPU memory profiler for tracking memory usage during training"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.snapshots = []
        self.reset()
    
    def reset(self):
        """Reset memory tracking"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        self.snapshots = []
        self.start_time = time.time()
    
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
        """Get memory breakdown by stage"""
        breakdown = defaultdict(int)
        for snapshot in self.snapshots:
            breakdown[snapshot.stage] = max(breakdown[snapshot.stage], snapshot.allocated)
        return dict(breakdown)


def prepare_snli_dataset(tokenizer, split="train", max_len=128):
    """Prepare SNLI dataset for training"""
    dataset = load_dataset("snli", split=split)
    dataset = dataset.filter(lambda x: x["label"] != -1)  # Remove unlabeled examples
    
    def tokenize_function(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    return tokenized_dataset


def prepare_model(
    token: str,
    is_lora: bool = False,
    lora_rank: int = 16,
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    profiler: GPUMemoryProfiler = None,
):
    """Prepare model and tokenizer"""
    if token:
        login(token)
    
    if profiler:
        profiler.snapshot("model_loading_start")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,  # SNLI has 3 labels
        torch_dtype=torch.float32,
        device_map="auto"
    )
    
    if profiler:
        profiler.snapshot("model_loading_end")
    
    # Apply LoRA if requested
    if is_lora:
        if profiler:
            profiler.snapshot("lora_setup_start")
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        model = get_peft_model(model, lora_config)
        
        if profiler:
            profiler.snapshot("lora_setup_end")
    
    return model, tokenizer


def profile_train_step(model, optimizer, criterion, batch, device, profiler: GPUMemoryProfiler):
    """Profile a single training step"""
    with profiler.profile_stage("forward_pass"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
    
    with profiler.profile_stage("backward_pass"):
        loss.backward()
    
    with profiler.profile_stage("optimizer_step"):
        optimizer.step()
        optimizer.zero_grad()
    
    return loss.item()


def run_memory_profile(
    token: str,
    device: torch.device,
    seq_length: int = 128,
    batch_size: int = 32,
    max_physical_batch_size: int = 1,
    is_lora: bool = False,
    lora_rank: int = 16,
    learning_rate: float = 1e-5,
    sigma: float = 1.0,
    max_grad_norm: float = 1.0,
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    num_steps: int = 5,
) -> MemoryProfileResult:
    """Run memory profiling for a specific configuration"""
    
    profiler = GPUMemoryProfiler(device)
    start_time = time.time()
    
    # Prepare model and tokenizer
    model, tokenizer = prepare_model(token, is_lora, lora_rank, model_name, profiler)
    
    # Prepare dataset
    with profiler.profile_stage("dataset_preparation"):
        dataset = prepare_snli_dataset(tokenizer, split="train[:1000]", max_len=seq_length)
        dataloader = DataLoader(dataset, batch_size=max_physical_batch_size, shuffle=True)
    
    # Setup optimizer and criterion
    with profiler.profile_stage("optimizer_setup"):
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
    
    # Setup privacy engine
    with profiler.profile_stage("privacy_engine_setup"):
        privacy_engine = PrivacyEngine()
        model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            epochs=1,
            target_epsilon=8.0,
            target_delta=1e-5,
            max_grad_norm=max_grad_norm,
        )
    
    # Setup batch memory manager if needed
    if batch_size > max_physical_batch_size:
        with profiler.profile_stage("batch_memory_manager_setup"):
            batch_memory_manager = BatchMemoryManager(
                data_loader=dataloader,
                max_physical_batch_size=max_physical_batch_size,
                optimizer=optimizer
            )
            dataloader = batch_memory_manager
    
    # Training loop
    model.train()
    losses = []
    
    with profiler.profile_stage("training_loop"):
        for step, batch in enumerate(dataloader):
            if step >= num_steps:
                break
            
            with profiler.profile_stage(f"training_step_{step}"):
                loss = profile_train_step(model, optimizer, criterion, batch, device, profiler)
                losses.append(loss)
    
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


def save_results(results: List[MemoryProfileResult], results_dir: str):
    """Save profiling results to files"""
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed results as JSON
    results_data = []
    for result in results:
        result_dict = {
            "seq_length": result.seq_length,
            "batch_size": result.batch_size,
            "max_physical_batch_size": result.max_physical_batch_size,
            "is_lora": result.is_lora,
            "lora_rank": result.lora_rank,
            "peak_memory": result.peak_memory,
            "memory_breakdown": result.memory_breakdown,
            "execution_time": result.execution_time,
            "snapshots": [
                {
                    "timestamp": s.timestamp,
                    "allocated": s.allocated,
                    "reserved": s.reserved,
                    "max_allocated": s.max_allocated,
                    "max_reserved": s.max_reserved,
                    "stage": s.stage,
                    "module": s.module,
                }
                for s in result.snapshots
            ],
        }
        results_data.append(result_dict)
    
    with open(os.path.join(results_dir, "memory_profile_results.json"), "w") as f:
        json.dump(results_data, f, indent=2)
    
    # Save summary as CSV
    summary_data = []
    for result in results:
        summary_data.append({
            "seq_length": result.seq_length,
            "batch_size": result.batch_size,
            "max_physical_batch_size": result.max_physical_batch_size,
            "is_lora": result.is_lora,
            "lora_rank": result.lora_rank,
            "peak_memory_gb": result.peak_memory / (1024**3),
            "execution_time": result.execution_time,
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(os.path.join(results_dir, "memory_profile_summary.csv"), index=False)
    
    print(f"Results saved to {results_dir}")
    print(f"Summary:\n{df}")


def create_visualizations(results: List[MemoryProfileResult], results_dir: str):
    """Create visualizations of memory usage"""
    os.makedirs(results_dir, exist_ok=True)
    
    # Memory usage over time
    plt.figure(figsize=(12, 8))
    for i, result in enumerate(results):
        timestamps = [s.timestamp for s in result.snapshots]
        allocated = [s.allocated / (1024**3) for s in result.snapshots]  # Convert to GB
        
        label = f"Seq={result.seq_length}, Batch={result.batch_size}, LoRA={result.is_lora}"
        plt.plot(timestamps, allocated, marker='o', label=label)
    
    plt.xlabel("Time (seconds)")
    plt.ylabel("Allocated Memory (GB)")
    plt.title("GPU Memory Usage Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "memory_usage_timeline.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Peak memory comparison
    if len(results) > 1:
        plt.figure(figsize=(10, 6))
        configs = []
        peak_memories = []
        
        for result in results:
            config = f"Seq={result.seq_length}\nBatch={result.batch_size}\nLoRA={result.is_lora}"
            configs.append(config)
            peak_memories.append(result.peak_memory / (1024**3))
        
        plt.bar(configs, peak_memories)
        plt.xlabel("Configuration")
        plt.ylabel("Peak Memory (GB)")
        plt.title("Peak Memory Usage by Configuration")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "peak_memory_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="Single GPU Memory Profiler for Opacus")
    parser.add_argument("--token", type=str, help="Hugging Face token")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", 
                       help="Model name to profile")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Logical batch size")
    parser.add_argument("--max_physical_batch_size", type=int, default=1, 
                       help="Maximum physical batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--sigma", type=float, default=1.0, help="Noise multiplier")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--num_steps", type=int, default=5, help="Number of training steps")
    parser.add_argument("--results_dir", type=str, default="./single_gpu_memory_results", 
                       help="Directory to save results")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA adaptation")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    
    args = parser.parse_args()
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("CUDA is not available. This profiler requires a GPU.")
        return
    
    device = torch.device("cuda:0")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(device).total_memory / (1024**3):.2f} GB")
    
    # Configuration to test
    config = {
        "token": args.token,
        "device": device,
        "seq_length": args.seq_length,
        "batch_size": args.batch_size,
        "max_physical_batch_size": args.max_physical_batch_size,
        "is_lora": args.use_lora,
        "lora_rank": args.lora_rank,
        "learning_rate": args.learning_rate,
        "sigma": args.sigma,
        "max_grad_norm": args.max_grad_norm,
        "model_name": args.model_name,
        "num_steps": args.num_steps,
    }
    
    print(f"Running memory profile with configuration:")
    for key, value in config.items():
        if key != "device":
            print(f"  {key}: {value}")
    
    try:
        result = run_memory_profile(**config)
        results = [result]
        
        # Save results and create visualizations
        save_results(results, args.results_dir)
        create_visualizations(results, args.results_dir)
        
        print(f"\nProfiling completed successfully!")
        print(f"Peak memory usage: {result.peak_memory / (1024**3):.2f} GB")
        print(f"Execution time: {result.execution_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error during profiling: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()