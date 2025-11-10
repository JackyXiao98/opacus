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

"""
Comprehensive benchmark comparing three DP-SGD algorithms:
1. Normal SGD (standard per-sample gradient computation)
2. Ghost Clipping (fast gradient clipping without Triton)
3. Flash Clipping (fast gradient clipping with Triton kernels)

This script measures both time and memory usage for each algorithm
using a 1B parameter model configuration.
"""

import argparse
import gc
import time
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Opacus imports
from opacus.grad_sample import GradSampleModule, GradSampleModuleFastGradientClipping
from opacus.optimizers import DPOptimizer, DPOptimizerFastGradientClipping
from opacus.utils.fast_gradient_clipping_utils import DPLossFastGradientClipping
from opacus.layers import DPMultiheadAttention
from opacus.grad_sample.triton_kernels import is_triton_available


class MemoryProfiler:
    """Memory profiler that works on both CPU and GPU"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.is_cuda = device == "cuda" and torch.cuda.is_available()
    
    def reset_stats(self):
        """Reset memory statistics"""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics in MB"""
        if not self.is_cuda:
            return {"allocated": 0, "reserved": 0, "max_allocated": 0}
        
        return {
            "allocated": torch.cuda.memory_allocated(self.device) / 2**20,
            "reserved": torch.cuda.memory_reserved(self.device) / 2**20,
            "max_allocated": torch.cuda.max_memory_allocated(self.device) / 2**20,
        }
    
    def get_peak_memory_mb(self) -> float:
        """Get peak memory usage in MB"""
        if not self.is_cuda:
            return 0.0
        return torch.cuda.max_memory_allocated(self.device) / 2**20


class DPCompatibleTransformerLayer(nn.Module):
    """Opacus-compatible Transformer layer without parameter tying"""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Use Opacus-compatible multihead attention
        self.self_attn = DPMultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x


class SimpleBigModel(nn.Module):
    """
    Custom model without parameter tying, compatible with Ghost/Flash Clipping.
    Uses Opacus-compatible Transformer architecture.
    """
    
    def __init__(self, vocab_size: int = 32000, hidden_dim: int = 512,
                 num_layers: int = 4, num_heads: int = 8, seq_len: int = 1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(seq_len, hidden_dim)
        
        # Transformer layers (Opacus compatible)
        self.layers = nn.ModuleList([
            DPCompatibleTransformerLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Output layer (NO parameter tying)
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        batch_size, seq_len = input_ids.shape
        
        # Create position ids
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeddings = self.embedding(input_ids)
        pos_embeddings = self.pos_embedding(position_ids)
        hidden_states = token_embeddings + pos_embeddings
        
        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # Final layer norm and projection
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape(-1, self.vocab_size), shift_labels.reshape(-1))
        
        return {"loss": loss, "logits": logits}
    
    def count_parameters(self) -> int:
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters())


def get_model_config(model_size: str = "1b") -> Dict[str, Any]:
    """Get model configuration for different sizes"""
    configs = {
        "tiny": {
            "vocab_size": 1000,
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "seq_len": 256
        },
        "small": {
            "vocab_size": 8000,
            "hidden_dim": 256,
            "num_layers": 4,
            "num_heads": 8,
            "seq_len": 512
        },
        "medium": {
            "vocab_size": 16000,
            "hidden_dim": 512,
            "num_layers": 8,
            "num_heads": 8,
            "seq_len": 1024
        },
        "1b": {
            "vocab_size": 32000,
            "hidden_dim": 2048,
            "num_layers": 16,
            "num_heads": 16,
            "seq_len": 4096
        },
        "3b": {
            "vocab_size": 32000,
            "hidden_dim": 3200,
            "num_layers": 26,
            "num_heads": 32,
            "seq_len": 2048
        },
        "test": {
            "vocab_size": 8,
            "hidden_dim": 8,
            "num_layers": 1,
            "num_heads": 1,
            "seq_len": 4096*8
        }
    }
    return configs.get(model_size, configs["1b"])


def get_random_dataloader(batch_size: int, seq_len: int, vocab_size: int,
                          device: str, num_batches: int = 10) -> DataLoader:
    """Generate random DataLoader for training"""
    input_ids = torch.randint(0, vocab_size, (num_batches * batch_size, seq_len), dtype=torch.long).contiguous()
    labels = torch.randint(0, vocab_size, (num_batches * batch_size, seq_len), dtype=torch.long).contiguous()
    
    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                           pin_memory=True if device == "cuda" else False)
    
    return dataloader


class TrainerBase(ABC):
    """Abstract base class for all trainers"""
    
    def __init__(self, model: nn.Module, optimizer_cls: type,
                 optimizer_params: Dict[str, Any], device: str):
        self.model = model
        self.optimizer_cls = optimizer_cls
        self.optimizer_params = optimizer_params
        self.device = device
        self.optimizer = None
        self.criterion = None
    
    @abstractmethod
    def setup_optimizer(self, dataloader: Optional[DataLoader] = None):
        """Setup optimizer and any privacy mechanisms"""
        pass
    
    def cleanup_memory(self):
        """Clean up memory and resources"""
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            self.optimizer.zero_grad()
        
        if hasattr(self, 'model') and self.model is not None:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad = None
        
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        gc.collect()
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """Execute one training step and return loss"""
        input_ids, labels = batch
        
        input_ids = input_ids.to(self.device, non_blocking=True).contiguous()
        labels = labels.to(self.device, non_blocking=True).contiguous()
        
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(input_ids, labels=labels)
        loss = outputs["loss"]
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        self.optimizer.step()
        
        return loss.item()


class NormalSGDTrainer(TrainerBase):
    """Normal SGD trainer with per-sample gradients"""
    
    def setup_optimizer(self, dataloader: Optional[DataLoader] = None):
        """Setup Normal SGD with GradSampleModule"""
        # Wrap with GradSampleModule for per-sample gradients
        self.model = GradSampleModule(self.model)
        
        base_optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_params)
        
        self.optimizer = DPOptimizer(
            optimizer=base_optimizer,
            noise_multiplier=0.0,  # No noise for fair comparison
            max_grad_norm=1.0,
            expected_batch_size=dataloader.batch_size if dataloader else 1,
        )


class GhostClippingTrainer(TrainerBase):
    """Ghost Clipping trainer (fast gradient clipping without Triton)"""
    
    def setup_optimizer(self, dataloader: Optional[DataLoader] = None):
        """Setup Ghost Clipping"""
        self.model = GradSampleModuleFastGradientClipping(
            self.model,
            batch_first=True,
            max_grad_norm=1.0,
            use_ghost_clipping=True,
            use_triton=False,  # No Triton
            loss_reduction="mean"
        )
        
        base_optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_params)
        
        self.optimizer = DPOptimizerFastGradientClipping(
            optimizer=base_optimizer,
            noise_multiplier=0.0,  # No noise for fair comparison
            max_grad_norm=1.0,
            expected_batch_size=dataloader.batch_size if dataloader else 1,
            loss_reduction="mean"
        )
        
        self.criterion = nn.CrossEntropyLoss(reduction="mean")


class FlashClippingTrainer(TrainerBase):
    """Flash Clipping trainer (fast gradient clipping with Triton)"""
    
    def setup_optimizer(self, dataloader: Optional[DataLoader] = None):
        """Setup Flash Clipping"""
        self.model = GradSampleModuleFastGradientClipping(
            self.model,
            batch_first=True,
            max_grad_norm=1.0,
            use_ghost_clipping=True,
            use_triton=True,  # Enable Triton
            loss_reduction="mean"
        )
        
        base_optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_params)
        
        self.optimizer = DPOptimizerFastGradientClipping(
            optimizer=base_optimizer,
            noise_multiplier=0.0,  # No noise for fair comparison
            max_grad_norm=1.0,
            expected_batch_size=dataloader.batch_size if dataloader else 1,
            loss_reduction="mean"
        )
        
        self.criterion = nn.CrossEntropyLoss(reduction="mean")


class AlgorithmBenchmark:
    """Benchmark different DP-SGD algorithms"""
    
    def __init__(self, device: str = "cpu", verbose: bool = True):
        self.device = device
        self.verbose = verbose
        self.profiler = MemoryProfiler(device)
    
    def benchmark_trainer(self, trainer_cls: type, model: nn.Module,
                         dataloader: DataLoader, trainer_name: str,
                         num_steps: int = 3, warmup_steps: int = 1) -> Dict:
        """Benchmark a single trainer"""
        
        print(f"\n{'='*60}")
        print(f"Benchmarking: {trainer_name}")
        print(f"{'='*60}")
        
        self.profiler.reset_stats()
        
        # Setup trainer
        optimizer_params = {"lr": 0.001}
        trainer = trainer_cls(
            model=model,
            optimizer_cls=optim.Adam,
            optimizer_params=optimizer_params,
            device=self.device
        )
        
        setup_start = time.perf_counter()
        trainer.setup_optimizer(dataloader)
        setup_time = time.perf_counter() - setup_start
        
        setup_memory = self.profiler.get_peak_memory_mb()
        self.profiler.reset_stats()
        
        print(f"Setup time: {setup_time:.4f}s")
        if self.device == "cuda":
            print(f"Setup memory: {setup_memory:.2f} MB")
        
        # Warmup
        print(f"\nWarming up ({warmup_steps} steps)...")
        data_iter = iter(dataloader)
        for step in range(warmup_steps):
            batch = next(data_iter, None)
            if batch is None:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            trainer.training_step(batch)
        
        # Benchmark
        print(f"\nBenchmarking ({num_steps} steps)...")
        times = []
        peak_memories = []
        
        data_iter = iter(dataloader)
        for step in range(num_steps):
            batch = next(data_iter, None)
            if batch is None:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            self.profiler.reset_stats()
            
            if self.device != "cpu":
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            loss = trainer.training_step(batch)
            
            if self.device != "cpu":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
            step_time = end_time - start_time
            peak_memory = self.profiler.get_peak_memory_mb()
            
            times.append(step_time)
            peak_memories.append(peak_memory)
            
            if self.verbose:
                if self.device == "cuda":
                    print(f"  Step {step+1}: {step_time:.4f}s, {peak_memory:.2f} MB, loss={loss:.4f}")
                else:
                    print(f"  Step {step+1}: {step_time:.4f}s, loss={loss:.4f}")
        
        # Calculate statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        mean_memory = np.mean(peak_memories) if self.device == "cuda" else 0
        std_memory = np.std(peak_memories) if self.device == "cuda" else 0
        
        results = {
            "algorithm": trainer_name,
            "mean_time": mean_time,
            "std_time": std_time,
            "mean_memory": mean_memory,
            "std_memory": std_memory,
            "times": times,
            "peak_memories": peak_memories,
            "setup_time": setup_time,
            "setup_memory": setup_memory,
        }
        
        print(f"\nResults for {trainer_name}:")
        print(f"  Time: {mean_time:.4f}±{std_time:.4f}s per step")
        if self.device == "cuda":
            print(f"  Memory: {mean_memory:.2f}±{std_memory:.2f} MB peak per step")
        
        # Cleanup
        trainer.cleanup_memory()
        del trainer
        
        return results
    
    def run_comprehensive_benchmark(self, model_size: str = "1b", batch_size: int = 1,
                                   seq_length: int = 128, num_steps: int = 3,
                                   warmup_steps: int = 1) -> List[Dict]:
        """Run comprehensive benchmark comparing all three algorithms"""
        
        print("="*70)
        print("COMPREHENSIVE ALGORITHM BENCHMARK")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Model size: {model_size}")
        print(f"Batch size: {batch_size}")
        print(f"Sequence length: {seq_length}")
        print(f"Number of steps: {num_steps}")
        
        if not is_triton_available():
            print("\nWARNING: Triton not available!")
            print("Flash clipping will fall back to ghost clipping.")
        
        # Get model configuration
        model_config = get_model_config(model_size)
        
        results = []
        
        # Test each algorithm
        trainer_configs = [
            (NormalSGDTrainer, "Normal SGD"),
            (GhostClippingTrainer, "Ghost Clipping"),
            (FlashClippingTrainer, "Flash Clipping"),
        ]
        
        for trainer_cls, trainer_name in trainer_configs:
            try:
                print(f"\nCreating model for {trainer_name}...")
                model = SimpleBigModel(**model_config).to(self.device)
                param_count = model.count_parameters()
                print(f"Model parameters: {param_count:,} ({param_count/1e6:.1f}M)")
                
                dataloader = get_random_dataloader(
                    batch_size=batch_size,
                    seq_len=seq_length,
                    vocab_size=model_config["vocab_size"],
                    device=self.device
                )
                
                result = self.benchmark_trainer(
                    trainer_cls, model, dataloader, trainer_name,
                    num_steps, warmup_steps
                )
                results.append(result)
                
                # Clean up
                del model
                del dataloader
                if self.device != "cpu":
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"\nError benchmarking {trainer_name}: {e}")
                traceback.print_exc()
                continue
        
        # Print summary
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results: List[Dict]):
        """Print comparison summary"""
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        
        if not results:
            print("No results to display.")
            return
        
        # Find baseline (Normal SGD)
        baseline = None
        for r in results:
            if r["algorithm"] == "Normal SGD":
                baseline = r
                break
        
        print(f"\n{'Algorithm':<20} {'Time (s)':<15} {'Memory (MB)':<15} {'Speedup':<10}")
        print("-"*70)
        
        for r in results:
            time_str = f"{r['mean_time']:.4f}±{r['std_time']:.4f}"
            
            if self.device == "cuda":
                mem_str = f"{r['mean_memory']:.2f}±{r['std_memory']:.2f}"
            else:
                mem_str = "N/A (CPU)"
            
            if baseline and baseline != r:
                speedup = baseline['mean_time'] / r['mean_time']
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "baseline"
            
            print(f"{r['algorithm']:<20} {time_str:<15} {mem_str:<15} {speedup_str:<10}")
        
        # Memory comparison
        if self.device == "cuda" and baseline:
            print("\nMemory Savings vs Normal SGD:")
            print("-"*70)
            for r in results:
                if r != baseline:
                    mem_savings = baseline['mean_memory'] - r['mean_memory']
                    mem_savings_pct = (mem_savings / baseline['mean_memory']) * 100
                    print(f"{r['algorithm']:<20} {mem_savings:+.2f} MB ({mem_savings_pct:+.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark and compare DP-SGD algorithms"
    )
    parser.add_argument("--model_size", choices=["tiny", "small", "medium", "1b", "3b", "test"],
                       default="test", help="Model size to use")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=4096*8, help="Sequence length")
    parser.add_argument("--num_steps", type=int, default=3, help="Number of benchmark steps")
    parser.add_argument("--warmup_steps", type=int, default=1, help="Number of warmup steps")
    parser.add_argument("--device", default="cuda", help="Device (cpu or cuda)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = "cpu"
    
    # Create benchmark
    benchmark = AlgorithmBenchmark(device=args.device, verbose=args.verbose)
    
    # Run benchmark
    results = benchmark.run_comprehensive_benchmark(
        model_size=args.model_size,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_steps=args.num_steps,
        warmup_steps=args.warmup_steps,
    )
    
    return results


if __name__ == "__main__":
    main()
