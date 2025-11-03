#!/usr/bin/env python3
"""
PyTorch Profiling Script for Opacus DP-SGD Performance Analysis

This script compares GPU memory usage and I/O costs between:
- Standard SGD
- DP-SGD with flat clipping
- DP-SGD with Ghost Clipping (per_layer)

Usage:
    python profiling_script.py --mode=profile  # Run full GPU profiling
    python profiling_script.py --mode=test     # Run local CPU test
"""

import argparse
import gc
import os
import random
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.profiler

# Try to import psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    print("Warning: psutil not available. Memory monitoring will be limited.")
    PSUTIL_AVAILABLE = False

try:
    import opacus
    from opacus import PrivacyEngine
    from opacus.grad_sample import GradSampleModule, GradSampleModuleFastGradientClipping
    from opacus.grad_sample.utils import wrap_model
    from opacus.optimizers import DPOptimizer, DPOptimizerFastGradientClipping
    from opacus.layers import DPMultiheadAttention
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    OPACUS_AVAILABLE = True
except ImportError:
    print("Warning: Opacus not available. DP-SGD trainers will not work.")
    OPACUS_AVAILABLE = False


def get_memory_usage(device: str = "cuda") -> Dict[str, float]:
    """Get current memory usage statistics"""
    memory_info = {}
    
    # GPU memory usage
    if device == "cuda" and torch.cuda.is_available():
        memory_info["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
        memory_info["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
        memory_info["gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2
        memory_info["gpu_max_reserved_mb"] = torch.cuda.max_memory_reserved() / 1024**2
    
    # System memory usage
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        memory_info["system_rss_mb"] = process.memory_info().rss / 1024**2
        memory_info["system_vms_mb"] = process.memory_info().vms / 1024**2
        memory_info["system_percent"] = process.memory_percent()
    
    return memory_info


def print_memory_usage(prefix: str, device: str = "cuda"):
    """Print current memory usage with a prefix"""
    memory_info = get_memory_usage(device)
    print(f"{prefix} Memory Usage:")
    
    if device == "cuda" and torch.cuda.is_available():
        print(f"  GPU Allocated: {memory_info.get('gpu_allocated_mb', 0):.1f} MB")
        print(f"  GPU Reserved: {memory_info.get('gpu_reserved_mb', 0):.1f} MB")
        print(f"  GPU Max Allocated: {memory_info.get('gpu_max_allocated_mb', 0):.1f} MB")
    
    if PSUTIL_AVAILABLE:
        print(f"  System RSS: {memory_info.get('system_rss_mb', 0):.1f} MB")
        print(f"  System Memory %: {memory_info.get('system_percent', 0):.1f}%")


def print_detailed_error(error: Exception, context: str = "", show_traceback: bool = True):
    """Print detailed error information with context"""
    print(f"\n{'='*60}")
    print(f"❌ ERROR in {context}")
    print(f"{'='*60}")
    print(f"Error Type: {type(error).__name__}")
    print(f"Error Message: {str(error)}")
    
    if show_traceback:
        print(f"\nDetailed Traceback:")
        print("-" * 40)
        traceback.print_exc()
        print("-" * 40)
    
    print(f"{'='*60}\n")


def safe_cleanup_objects(trainer=None, model=None, dataloader=None, device="cuda"):
    """Safely clean up objects and memory without raising exceptions"""
    cleanup_errors = []
    
    # Clean up trainer components
    if trainer is not None:
        for attr_name in ['model', 'optimizer', 'privacy_engine', 'criterion']:
            if hasattr(trainer, attr_name):
                try:
                    delattr(trainer, attr_name)
                except Exception as e:
                    cleanup_errors.append(f"Failed to delete trainer.{attr_name}: {e}")
        
        try:
            del trainer
        except Exception as e:
            cleanup_errors.append(f"Failed to delete trainer: {e}")
    
    # Clean up model and dataloader
    for obj_name, obj in [('model', model), ('dataloader', dataloader)]:
        if obj is not None:
            try:
                del obj
            except Exception as e:
                cleanup_errors.append(f"Failed to delete {obj_name}: {e}")
    
    # GPU memory cleanup
    if device == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as e:
            cleanup_errors.append(f"Failed to clear CUDA cache: {e}")
    
    # Garbage collection
    try:
        gc.collect()
    except Exception as e:
        cleanup_errors.append(f"Failed to run garbage collection: {e}")
    
    # Report cleanup errors if any
    if cleanup_errors:
        print("⚠️  Cleanup warnings:")
        for error in cleanup_errors:
            print(f"  - {error}")
    
    return len(cleanup_errors) == 0


class DPCompatibleTransformerLayer(nn.Module):
    """Opacus 兼容的 Transformer 层"""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 使用 Opacus 兼容的多头注意力
        if OPACUS_AVAILABLE:
            self.self_attn = DPMultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        else:
            # 如果 Opacus 不可用，使用标准的多头注意力
            self.self_attn = nn.MultiheadAttention(
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
    A simple but scalable model that can be configured for different sizes.
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
        
        # Output layer
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


def get_model_config(model_size: str = "small") -> Dict[str, Any]:
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
            "num_layers": 24,
            "num_heads": 32,
            "seq_len": 2048
        }
    }
    return configs.get(model_size, configs["small"])


def get_random_dataloader(batch_size: int, seq_len: int, vocab_size: int, 
                         device: str, num_batches: int = 10) -> DataLoader:
    """Generate random DataLoader for training"""
    # Generate random data and ensure contiguity for Opacus compatibility
    input_ids = torch.randint(0, vocab_size, (num_batches * batch_size, seq_len), dtype=torch.long).contiguous()
    labels = torch.randint(0, vocab_size, (num_batches * batch_size, seq_len), dtype=torch.long).contiguous()
    
    # Create dataset and dataloader
    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True if device == "cuda" else False)
    
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
        self.privacy_engine = None
        
    @abstractmethod
    def setup_optimizer(self, dataloader: Optional[DataLoader] = None):
        """Setup optimizer and any privacy mechanisms"""
        pass
    
    def cleanup_memory(self):
        """Clean up memory and resources after training"""
        # Clear gradients
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            self.optimizer.zero_grad()
        
        # Clear model gradients
        if hasattr(self, 'model') and self.model is not None:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad = None
        
        # Clear CUDA cache if using GPU
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
    
    def profile_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """Execute one training step and return loss"""
        input_ids, labels = batch
        
        # Ensure tensors are properly moved to device and contiguous
        input_ids = input_ids.to(self.device, non_blocking=True).contiguous()
        labels = labels.to(self.device, non_blocking=True).contiguous()
        
        # Clear gradients before forward pass
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(input_ids, labels=labels)
        loss = outputs["loss"]
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        self.optimizer.step()
        
        return loss.item()
    
    def run_profiling(self, dataloader: DataLoader, log_name: str, 
                     num_steps: int = 6) -> None:
        """Run profiling with torch.profiler"""
        # Setup optimizer (let errors propagate naturally)
        self.setup_optimizer(dataloader)
        
        # Run profiling with automatic cleanup
        try:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./runs/{log_name}'),
                profile_memory=True,  # Key: Enable memory profiling
                record_shapes=True,
                with_stack=True
            ) as prof:
                data_iter = iter(dataloader)
                for step in range(num_steps):
                    # Handle data iteration naturally - restart if needed
                    if step > 0 and step % len(dataloader) == 0:
                        data_iter = iter(dataloader)
                    
                    batch = next(data_iter, None)
                    if batch is None:
                        data_iter = iter(dataloader)
                        batch = next(data_iter)
                    
                    loss = self.profile_step(batch)
                    prof.step()
                    
                    if step % 2 == 0:
                        print(f"Step {step}, Loss: {loss:.4f}")
        
        finally:
            # Always clean up memory after profiling
            self.cleanup_memory()


class StandardTrainer(TrainerBase):
    """Standard SGD trainer without differential privacy"""
    
    def setup_optimizer(self, dataloader: Optional[DataLoader] = None):
        """Setup standard SGD optimizer"""
        self.optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_params)


class DPSGDTrainer(TrainerBase):
    """DP-SGD trainer with flat clipping (standard per-sample gradients)"""
    
    def setup_optimizer(self, dataloader: Optional[DataLoader] = None):
        """Setup DP-SGD with GradSampleModule and DPOptimizer"""
        if not OPACUS_AVAILABLE:
            raise ImportError("Opacus is required for DP-SGD training")
        
        # Wrap model with GradSampleModule for per-sample gradients
        self.model = GradSampleModule(self.model)
        
        # Create standard optimizer
        base_optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_params)
        
        # Wrap with DPOptimizer for differential privacy
        self.optimizer = DPOptimizer(
            optimizer=base_optimizer,
            noise_multiplier=1.0,  # Fixed noise multiplier for consistent comparison
            max_grad_norm=1.0,
            expected_batch_size=dataloader.batch_size if dataloader else 1,
        )


class DPGhostClippingTrainer(TrainerBase):
    """DP-SGD trainer with Ghost Clipping (memory-efficient gradient computation)"""
    
    def setup_optimizer(self, dataloader: Optional[DataLoader] = None):
        """Setup DP-SGD with Ghost Clipping using PrivacyEngine"""
        if not OPACUS_AVAILABLE:
            raise ImportError("Opacus is required for DP-SGD training")
        
        # Create standard optimizer first
        base_optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_params)
        
        # Create PrivacyEngine with Ghost Clipping
        self.privacy_engine = PrivacyEngine()
        
        # Use Ghost Clipping mode - this is the real memory-efficient implementation
        # Ghost mode returns 4 values: (model, optimizer, criterion, dataloader)
        self.model, self.optimizer, self.criterion, _ = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=base_optimizer,
            data_loader=dataloader,
            epochs=1,  # We're only doing profiling, so 1 epoch is fine
            target_epsilon=8.0,  # Standard epsilon for comparison
            target_delta=1e-5,   # Standard delta
            max_grad_norm=1.0,   # Single value for ghost clipping
            grad_sample_mode="ghost",  # This is the key - use ghost mode for memory efficiency
            clipping="flat"  # Use flat clipping with ghost mode
        )
    
    def profile_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """Execute one training step - let errors propagate naturally with context"""
        # Let the parent method handle the training step
        # If there are tensor compatibility issues, they will be caught at a higher level
        # and reported with full context
        return super().profile_step(batch)


class DPFastGradientClippingTrainer(TrainerBase):
    """DP-SGD trainer with Fast Gradient Clipping (memory-efficient gradient clipping without storing per-sample gradients)"""
    
    def setup_optimizer(self, dataloader: Optional[DataLoader] = None):
        """Setup DP-SGD with Fast Gradient Clipping using GradSampleModuleFastGradientClipping"""
        if not OPACUS_AVAILABLE:
            raise ImportError("Opacus is required for DP-SGD training")
        
        # Wrap model with GradSampleModuleFastGradientClipping for efficient gradient computation
        # This computes gradient norms without storing full per-sample gradients
        self.model = GradSampleModuleFastGradientClipping(
            self.model,
            batch_first=True,
            loss_reduction="mean",
            max_grad_norm=1.0,
            use_ghost_clipping=False  # Use Fast Gradient Clipping, not Ghost Clipping
        )
        
        # Create standard optimizer
        base_optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_params)
        
        # Wrap with DPOptimizerFastGradientClipping for differential privacy
        self.optimizer = DPOptimizerFastGradientClipping(
            optimizer=base_optimizer,
            noise_multiplier=1.0,  # Fixed noise multiplier for consistent comparison
            max_grad_norm=1.0,
            expected_batch_size=dataloader.batch_size if dataloader else 4,
            loss_reduction="mean"
        )


def run_full_profile():
    """Run full profiling experiment on GPU with large model (~1B parameters)"""
    print("Starting full GPU profiling with ~1B parameter model...")
    
    if not torch.cuda.is_available():
        print("CUDA not available! Switching to CPU for testing...")
        device = "cpu"
        model_size = "small"
    else:
        device = "cuda"
        model_size = "1b"
    
    # Initial memory usage
    print_memory_usage("Initial", device)
    
    # Experiment configurations
    TRAINER_CLASSES = [DPGhostClippingTrainer, StandardTrainer, DPSGDTrainer, DPFastGradientClippingTrainer]
    BATCH_SIZES = [4, 8] if device == "cuda" else [2, 4]  # Smaller batches for 1B model
    SEQ_LENGTHS = [512, 1024] if device == "cuda" else [256, 512]
    
    # Get model config
    model_config = get_model_config(model_size)
    print(f"Using model size: {model_size}")
    print(f"Model config: {model_config}")
    
    for trainer_cls in TRAINER_CLASSES:
        trainer_name = trainer_cls.__name__
        print(f"\n=== Testing {trainer_name} ===")
        
        # Skip DP trainers if Opacus not available
        if not OPACUS_AVAILABLE and "DP" in trainer_name:
            print(f"Skipping {trainer_name} - Opacus not available")
            continue
        
        for batch_size in BATCH_SIZES:
            for seq_len in SEQ_LENGTHS:
                log_name = f"{trainer_name}_bs{batch_size}_seq{seq_len}"
                print(f"\nRunning {trainer_name} with batch_size={batch_size}, seq_len={seq_len}")
                
                # Initialize variables for cleanup
                trainer = None
                model = None
                dataloader = None
                
                # Clear GPU cache before starting
                if device == "cuda":
                    torch.cuda.empty_cache()
                
                # Report memory before model creation
                print_memory_usage("Before model creation", device)
                
                try:
                    # Create fresh model and data
                    model = SimpleBigModel(**model_config).to(device)
                    param_count = model.count_parameters()
                    print(f"Model parameters: {param_count:,} ({param_count/1e6:.1f}M)")
                    
                    # Report memory after model creation
                    print_memory_usage("After model creation", device)
                    
                    dataloader = get_random_dataloader(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        vocab_size=model_config["vocab_size"],
                        device=device
                    )
                    
                    # Setup trainer
                    optimizer_params = {"lr": 0.001, "momentum": 0.9}
                    trainer = trainer_cls(
                        model=model,
                        optimizer_cls=optim.SGD,
                        optimizer_params=optimizer_params,
                        device=device
                    )
                    
                    # Run profiling (errors will propagate naturally)
                    trainer.run_profiling(dataloader, log_name)
                    print(f"✓ Completed profiling for {log_name}")
                    
                    # Report memory after profiling
                    print_memory_usage("After profiling", device)
                    
                except Exception as e:
                    # Print detailed error information
                    print_detailed_error(e, f"profiling {log_name}")
                    print(f"⏭️  Skipping this configuration and continuing...")
                
                # Always perform cleanup (no try-catch needed)
                safe_cleanup_objects(trainer, model, dataloader, device)
                print_memory_usage("After cleanup", device)
        
        # Additional cleanup after each trainer class is completed
        print(f"Completed all configurations for {trainer_name}, performing final cleanup...")
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    print("\n=== Full profiling completed ===")
    print("View results with: tensorboard --logdir=./runs")


def run_local_test():
    """Run local test on CPU with small model (~1M parameters)"""
    print("Starting local CPU test with small model...")
    
    device = "cpu"
    model_config = get_model_config("tiny")  # Very small model for testing
    
    TRAINER_CLASSES = [StandardTrainer, DPSGDTrainer, DPGhostClippingTrainer, DPFastGradientClippingTrainer]
    
    for trainer_cls in TRAINER_CLASSES:
        trainer_name = trainer_cls.__name__
        print(f"\n=== Testing {trainer_name} ===")
        
        # Skip DP trainers if Opacus not available
        if not OPACUS_AVAILABLE and "DP" in trainer_name:
            print(f"[TEST SKIP] {trainer_name} - Opacus not available")
            continue
        
        # Initialize variables for cleanup
        model = None
        trainer = None
        
        try:
            # Create model
            model = SimpleBigModel(**model_config).to(device)
            param_count = model.count_parameters()
            print(f"Model parameters: {param_count:,} ({param_count/1e6:.2f}M)")
            
            # Create data
            dataloader = get_random_dataloader(
                batch_size=2,
                seq_len=128,
                vocab_size=model_config["vocab_size"],
                device=device,
                num_batches=2
            )
            
            # Setup trainer
            optimizer_params = {"lr": 0.001, "momentum": 0.9}
            trainer = trainer_cls(
                model=model,
                optimizer_cls=optim.SGD,
                optimizer_params=optimizer_params,
                device=device
            )
            
            # Setup optimizer (let errors propagate)
            trainer.setup_optimizer(dataloader)
            
            # Run one step (no profiling, let errors propagate)
            batch = next(iter(dataloader))
            loss = trainer.profile_step(batch)
            
            print(f"[TEST OK] {trainer_name} passed local test. Loss: {loss:.4f}")
            
        except Exception as e:
            print_detailed_error(e, f"testing {trainer_name}", show_traceback=False)
            print(f"[TEST FAIL] {trainer_name} failed")
        
        # Clean up after each test
        safe_cleanup_objects(trainer, model, device=device)
    
    print("\n=== Local testing completed ===")


def run_single_config(trainer_name: str, batch_size: int, seq_len: int, model_size: str = "1b"):
    """Run profiling for a single configuration"""
    print(f"Starting single configuration profiling...")
    print(f"Trainer: {trainer_name}, Batch Size: {batch_size}, Seq Length: {seq_len}, Model: {model_size}")
    
    # Determine device
    if not torch.cuda.is_available():
        print("CUDA not available! Switching to CPU for testing...")
        device = "cpu"
        model_size = "small"  # Use smaller model for CPU
    else:
        device = "cuda"
    
    # Get model config and trainer class
    model_config = get_model_config(model_size)
    trainer_classes = {
        "StandardTrainer": StandardTrainer,
        "DPSGDTrainer": DPSGDTrainer, 
        "DPGhostClippingTrainer": DPGhostClippingTrainer,
        "DPFastGradientClippingTrainer": DPFastGradientClippingTrainer
    }
    
    if trainer_name not in trainer_classes:
        raise ValueError(f"Unknown trainer: {trainer_name}. Available: {list(trainer_classes.keys())}")
    
    trainer_cls = trainer_classes[trainer_name]
    
    # Skip DP trainers if Opacus not available
    if not OPACUS_AVAILABLE and "DP" in trainer_name:
        print(f"Skipping {trainer_name} - Opacus not available")
        return
    
    # Initial memory usage
    print_memory_usage("Initial", device)
    
    # Initialize variables for cleanup
    trainer = None
    model = None
    dataloader = None
    
    # Clear GPU cache before starting
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # Report memory before model creation
    print_memory_usage("Before model creation", device)
    
    try:
        # Create fresh model and data
        model = SimpleBigModel(**model_config).to(device)
        param_count = model.count_parameters()
        print(f"Model parameters: {param_count:,} ({param_count/1e6:.1f}M)")
        
        # Report memory after model creation
        print_memory_usage("After model creation", device)
        
        dataloader = get_random_dataloader(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=model_config["vocab_size"],
            device=device
        )
        
        # Setup trainer
        optimizer_params = {"lr": 0.001, "momentum": 0.9}
        trainer = trainer_cls(
            model=model,
            optimizer_cls=optim.SGD,
            optimizer_params=optimizer_params,
            device=device
        )
        
        # Generate log name
        log_name = f"{trainer_name}_bs{batch_size}_seq{seq_len}"
        
        # Run profiling (errors will propagate naturally)
        trainer.run_profiling(dataloader, log_name)
        print(f"✓ Completed profiling for {log_name}")
        
        # Report memory after profiling
        print_memory_usage("After profiling", device)
        
    except Exception as e:
        # Print detailed error information
        print_detailed_error(e, f"profiling {trainer_name}")
        raise  # Re-raise to indicate failure to shell script
    
    finally:
        # Always perform cleanup
        safe_cleanup_objects(trainer, model, dataloader, device)
        print_memory_usage("After cleanup", device)
        
        # Final cleanup
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    print(f"✓ Single configuration profiling completed successfully")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="PyTorch Profiling Script for Opacus DP-SGD")
    parser.add_argument("--mode", type=str, default="profile", 
                       choices=["profile", "test", "single"],
                       help="Mode: 'profile' for full GPU profiling, 'test' for local CPU test, 'single' for single configuration")
    
    # Single configuration arguments
    parser.add_argument("--trainer", type=str, 
                       choices=["StandardTrainer", "DPSGDTrainer", "DPGhostClippingTrainer", "DPFastGradientClippingTrainer"],
                       help="Trainer class name for single mode")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for single mode")
    parser.add_argument("--seq-len", type=int, default=512,
                       help="Sequence length for single mode")
    parser.add_argument("--model-size", type=str, default="1b",
                       choices=["tiny", "small", "medium", "1b"],
                       help="Model size for single mode")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    if args.mode == "profile":
        run_full_profile()
    elif args.mode == "test":
        run_local_test()
    elif args.mode == "single":
        if not args.trainer:
            print("Error: --trainer is required for single mode")
            parser.print_help()
            exit(1)
        run_single_config(args.trainer, args.batch_size, args.seq_len, args.model_size)
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()


"""
=== 分析指南 (Analysis Guide) ===

启动 TensorBoard 查看结果:
    tensorboard --logdir=./runs

1. 如何分析显存 (Memory Cost):
   在 TensorBoard 的 'PyTorch Profiler' 插件中:
   
   a) 进入 'Memory View' 标签页
   b) 查看以下关键指标:
      - 'Self CUDA Memory': 每个操作的直接显存使用量
      - 'CUDA Memory Usage': 总体显存使用趋势图
      - 'Memory Timeline': 显存分配和释放的时间线
   
   c) 重点对比:
      - StandardTrainer vs DPSGDTrainer: DP-SGD 由于需要存储每个样本的梯度，显存使用量会显著增加
      - DPSGDTrainer vs DPGhostClippingTrainer: Ghost Clipping 应该显著减少显存占用
   
   d) 关注的操作类型:
      - 'aten::*' 操作中与梯度计算相关的部分
      - 'autograd::*' 操作的内存分配
      - 'Optimizer.step' 相关的内存使用

2. 如何分析 I/O Cost:
   在 'Trace Viewer' 或 'Kernel' 视图中:
   
   a) 查找关键的 CUDA Kernel 名称:
      - 'elementwise_kernel': 逐元素操作，如梯度裁剪中的 norm 计算
      - 'reduce_kernel': 归约操作，如计算梯度范数
      - 'gemm' 或 'cutlass': 矩阵乘法操作
      - 'cudnn_*': cuDNN 相关操作
      - 包含 'clip', 'norm', 'einsum' 的 kernel 名称
   
   b) 关注的性能指标:
      - 'Duration': kernel 执行时间，DP-SGD 应该显示更长的执行时间
      - 'Grid Size' 和 'Block Size': 并行度指标
      - 'Registers Per Thread': 寄存器使用量
      - 'Shared Memory': 共享内存使用量
   
   c) I/O 带宽分析:
      - 查看 GPU 利用率图表中的内存带宽使用
      - 对比不同训练方法的数据传输模式
      - DP-SGD 由于需要处理更多的梯度数据，应该显示更高的内存带宽需求
   
   d) 性能瓶颈识别:
      - 查找执行时间最长的 kernel
      - 识别内存带宽受限的操作
      - 对比 per-sample gradient clipping 与 Ghost Clipping 的性能差异

3. 预期的性能差异:
   - Standard SGD: 最低的显存和计算开销
   - DP-SGD (flat clipping): 显著增加的显存使用和 I/O 开销
   - DP-SGD (Ghost Clipping): 相比 flat clipping 减少的显存使用，但仍高于标准 SGD

通过这些分析，你可以量化 per-sample gradient clipping 带来的性能开销，
并验证 Ghost Clipping 等优化技术的效果。
"""