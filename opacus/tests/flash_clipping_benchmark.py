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

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple

from opacus.grad_sample import GradSampleModuleFastGradientClipping
from opacus.optimizers import DPOptimizerFastGradientClipping
from opacus.utils.fast_gradient_clipping_utils import DPLossFastGradientClipping
from opacus.utils.per_sample_gradients_utils import clone_module
from opacus.grad_sample.triton_kernels import is_triton_available


class BenchmarkDataset(Dataset):
    """Large synthetic dataset for benchmarking"""
    def __init__(self, size, seq_length, input_dim, num_classes=10):
        self.size = size
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Generate synthetic data
        self.sequences = torch.randn(self.size, self.seq_length, self.input_dim)
        self.labels = torch.randint(0, self.num_classes, (self.size,))

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.sequences[index], self.labels[index]


class BenchmarkModel(nn.Module):
    """Model with multiple linear layers for comprehensive benchmarking"""
    def __init__(self, input_dim=512, hidden_dims=[1024, 512, 256], num_classes=10):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        x = self.network(x)  # Apply to each timestep
        x = x.mean(dim=1)  # Global average pooling
        return x


class FlashClippingBenchmark:
    """Benchmark flash clipping against ghost clipping"""
    
    def __init__(self, device="cpu"):
        self.device = device
        # Create separate criterion instances for ghost and flash clipping
        self.ghost_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.flash_criterion = nn.CrossEntropyLoss(reduction="mean")
        
    def create_benchmark_setup(self, 
                             batch_size: int,
                             seq_length: int, 
                             input_dim: int,
                             hidden_dims: List[int],
                             num_classes: int = 10,
                             max_grad_norm: float = 1.0) -> Tuple:
        """Create models and data for benchmarking"""
        
        # Create dataset
        dataset = BenchmarkDataset(
            size=batch_size * 10,  # Multiple batches for thorough testing
            seq_length=seq_length,
            input_dim=input_dim,
            num_classes=num_classes
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Create base model
        base_model = BenchmarkModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes
        ).to(self.device)
        
        # Clone models
        ghost_model = clone_module(base_model).to(self.device)
        flash_model = clone_module(base_model).to(self.device)
        
        # Verify device consistency
        assert next(ghost_model.parameters()).device.type == self.device.split(':')[0] if ':' in self.device else self.device, \
            f"Ghost model not on correct device. Expected: {self.device}, Got: {next(ghost_model.parameters()).device}"
        assert next(flash_model.parameters()).device.type == self.device.split(':')[0] if ':' in self.device else self.device, \
            f"Flash model not on correct device. Expected: {self.device}, Got: {next(flash_model.parameters()).device}"
        
        return dataloader, ghost_model, flash_model, max_grad_norm
    
    def setup_ghost_clipping(self, model, max_grad_norm: float, batch_size: int):
        """Setup ghost clipping"""
        gsm = GradSampleModuleFastGradientClipping(
            model,
            batch_first=True,
            max_grad_norm=max_grad_norm,
            use_ghost_clipping=True,
            use_triton=False,
            loss_reduction="mean"
        )
        
        optimizer = torch.optim.Adam(gsm.parameters(), lr=0.001)
        dp_optimizer = DPOptimizerFastGradientClipping(
            optimizer,
            noise_multiplier=0.0,
            max_grad_norm=max_grad_norm,
            expected_batch_size=batch_size,
            loss_reduction="mean"
        )
        
        criterion = DPLossFastGradientClipping(gsm, dp_optimizer, self.ghost_criterion, loss_reduction="mean")
        
        return gsm, dp_optimizer, criterion
    
    def setup_flash_clipping(self, model, max_grad_norm: float, batch_size: int):
        """Setup flash clipping"""
        gsm = GradSampleModuleFastGradientClipping(
            model,
            batch_first=True,
            max_grad_norm=max_grad_norm,
            use_ghost_clipping=True,
            use_triton=True,
            loss_reduction="mean"
        )
        
        optimizer = torch.optim.Adam(gsm.parameters(), lr=0.001)
        dp_optimizer = DPOptimizerFastGradientClipping(
            optimizer,
            noise_multiplier=0.0,
            max_grad_norm=max_grad_norm,
            expected_batch_size=batch_size,
            loss_reduction="mean"
        )
        
        criterion = DPLossFastGradientClipping(gsm, dp_optimizer, self.flash_criterion, loss_reduction="mean")
        
        return gsm, dp_optimizer, criterion
    
    def benchmark_single_config(self, 
                              batch_size: int,
                              seq_length: int,
                              input_dim: int,
                              hidden_dims: List[int],
                              num_batches: int = 5,
                              warmup_batches: int = 2) -> Dict:
        """Benchmark a single configuration"""
        
        print(f"\nBenchmarking: batch_size={batch_size}, seq_length={seq_length}, "
              f"input_dim={input_dim}, hidden_dims={hidden_dims}")
        
        # Reset criterion instances for each configuration to avoid state pollution
        self.ghost_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.flash_criterion = nn.CrossEntropyLoss(reduction="mean")
        
        # Setup
        dataloader, ghost_model, flash_model, max_grad_norm = self.create_benchmark_setup(
            batch_size, seq_length, input_dim, hidden_dims
        )
        
        ghost_gsm, ghost_optimizer, ghost_criterion = self.setup_ghost_clipping(
            ghost_model, max_grad_norm, batch_size
        )
        
        flash_gsm, flash_optimizer, flash_criterion = self.setup_flash_clipping(
            flash_model, max_grad_norm, batch_size
        )
        
        # Warmup
        print("  Warming up...")
        for i, (data, target) in enumerate(dataloader):
            if i >= warmup_batches:
                break
            data, target = data.to(self.device), target.to(self.device)
            
            # Ghost warmup
            ghost_optimizer.zero_grad()
            ghost_gsm.train()
            loss = ghost_criterion(ghost_gsm(data), target)
            loss.backward()  # Trigger gradient computation and _norm_sample calculation
            
            # Flash warmup
            flash_optimizer.zero_grad()
            flash_gsm.train()
            loss = flash_criterion(flash_gsm(data), target)
            loss.backward()  # Trigger gradient computation and _norm_sample calculation
        
        # Benchmark ghost clipping
        print("  Benchmarking ghost clipping...")
        torch.cuda.synchronize() if self.device != "cpu" else None
        ghost_times = []
        
        for i, (data, target) in enumerate(dataloader):
            if i >= num_batches:
                break
            data, target = data.to(self.device), target.to(self.device)
            
            ghost_optimizer.zero_grad()
            ghost_gsm.train()
            
            start_time = time.perf_counter()
            loss = ghost_criterion(ghost_gsm(data), target)
            loss.backward()  # Trigger gradient computation and _norm_sample calculation
            torch.cuda.synchronize() if self.device != "cpu" else None
            end_time = time.perf_counter()
            
            ghost_times.append(end_time - start_time)
        
        # Benchmark flash clipping
        print("  Benchmarking flash clipping...")
        torch.cuda.synchronize() if self.device != "cpu" else None
        flash_times = []
        
        for i, (data, target) in enumerate(dataloader):
            if i >= num_batches:
                break
            data, target = data.to(self.device), target.to(self.device)
            
            flash_optimizer.zero_grad()
            flash_gsm.train()
            
            start_time = time.perf_counter()
            loss = flash_criterion(flash_gsm(data), target)
            loss.backward()  # Trigger gradient computation and _norm_sample calculation
            torch.cuda.synchronize() if self.device != "cpu" else None
            end_time = time.perf_counter()
            
            flash_times.append(end_time - start_time)
        
        # Calculate statistics
        ghost_mean = np.mean(ghost_times)
        ghost_std = np.std(ghost_times)
        flash_mean = np.mean(flash_times)
        flash_std = np.std(flash_times)
        
        speedup = ghost_mean / flash_mean if flash_mean > 0 else 0
        
        results = {
            "batch_size": batch_size,
            "seq_length": seq_length,
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "ghost_time_mean": ghost_mean,
            "ghost_time_std": ghost_std,
            "flash_time_mean": flash_mean,
            "flash_time_std": flash_std,
            "speedup": speedup,
            "ghost_times": ghost_times,
            "flash_times": flash_times
        }
        
        print(f"  Ghost clipping: {ghost_mean:.4f}±{ghost_std:.4f}s")
        print(f"  Flash clipping: {flash_mean:.4f}±{flash_std:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")
        
        return results
    
    def run_comprehensive_benchmark(self, force_run_without_triton: bool = False) -> List[Dict]:
        """Run comprehensive benchmark across different configurations
        
        Args:
            force_run_without_triton: If True, run benchmark even when Triton is not available
        """
        
        if not is_triton_available():
            print("WARNING: Triton not available!")
            print("Flash clipping will fall back to standard computation methods.")
            print("This may result in reduced performance benefits.")
            print("To install Triton: pip install triton")
            
            if not force_run_without_triton:
                print("Skipping benchmark to avoid misleading results.")
                print("Use force_run_without_triton=True to run anyway.")
                return []
            else:
                print("Continuing with fallback methods (results may not show expected speedup).")
                print("=" * 70)
        
        print("Running Flash Clipping Benchmark")
        print("=" * 50)
        
        # Different configurations to test
        configs = [
            # Small models
            {"batch_size": 32, "seq_length": 64, "input_dim": 128, "hidden_dims": [256, 128]},
            {"batch_size": 64, "seq_length": 64, "input_dim": 128, "hidden_dims": [256, 128]},
            
            # Medium models
            {"batch_size": 32, "seq_length": 128, "input_dim": 256, "hidden_dims": [512, 256, 128]},
            {"batch_size": 64, "seq_length": 128, "input_dim": 256, "hidden_dims": [512, 256, 128]},
            
            # Large models (if memory allows)
            {"batch_size": 16, "seq_length": 256, "input_dim": 512, "hidden_dims": [1024, 512, 256]},
            {"batch_size": 32, "seq_length": 256, "input_dim": 512, "hidden_dims": [1024, 512, 256]},
        ]
        
        results = []
        
        for config in configs:
            try:
                result = self.benchmark_single_config(**config)
                results.append(result)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  Skipping config due to OOM: {config}")
                    continue
                else:
                    raise e
        
        # Print summary
        print("\n" + "=" * 50)
        print("BENCHMARK SUMMARY")
        print("=" * 50)
        
        for result in results:
            print(f"Batch: {result['batch_size']}, Seq: {result['seq_length']}, "
                  f"Input: {result['input_dim']}, Hidden: {result['hidden_dims']}")
            print(f"  Speedup: {result['speedup']:.2f}x")
        
        avg_speedup = np.mean([r['speedup'] for r in results])
        print(f"\nAverage speedup: {avg_speedup:.2f}x")
        
        return results


def main():
    """Run the benchmark"""
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    benchmark = FlashClippingBenchmark(device=device)
    results = benchmark.run_comprehensive_benchmark()
    
    return results


if __name__ == "__main__":
    main()