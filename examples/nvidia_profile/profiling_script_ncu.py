#!/usr/bin/env python3
"""
NVIDIA NCU Profiling Script for Opacus DP-SGD Performance Analysis

This script compares GPU kernel performance and I/O costs between:
- Standard SGD
- DP-SGD with flat clipping
- DP-SGD with Ghost Clipping (per_layer)

Usage:
    python profiling_script_ncu.py --mode=profile --trainer=standard --batch_size=8 --seq_len=256
    python profiling_script_ncu.py --mode=test  # Run local CPU test
"""

import argparse
import random
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    import opacus
    from opacus import PrivacyEngine
    from opacus.grad_sample import GradSampleModule
    from opacus.grad_sample.utils import wrap_model
    from opacus.optimizers import DPOptimizer
    from opacus.layers import DPMultiheadAttention
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    OPACUS_AVAILABLE = True
except ImportError:
    print("Warning: Opacus not available. DP-SGD trainers will not work.")
    OPACUS_AVAILABLE = False


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
    input_ids = torch.randint(0, vocab_size, (num_batches * batch_size, seq_len)).contiguous()
    labels = torch.randint(0, vocab_size, (num_batches * batch_size, seq_len)).contiguous()
    
    # Move to device
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    
    # Create dataset and dataloader
    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
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
    
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """Execute one training step and return loss (renamed from profile_step)"""
        input_ids, labels = batch
        input_ids = input_ids.to(self.device).contiguous()
        labels = labels.to(self.device).contiguous()
        
        # Forward pass
        outputs = self.model(input_ids, labels=labels)
        loss = outputs["loss"]
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()


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
    """DP-SGD trainer with Ghost Clipping (memory-efficient per-layer clipping)"""
    
    def setup_optimizer(self, dataloader: Optional[DataLoader] = None):
        """Setup DP-SGD with Ghost Clipping using wrap_model"""
        if not OPACUS_AVAILABLE:
            raise ImportError("Opacus is required for DP-SGD training")
        
        # Wrap model with Ghost Clipping mode for memory efficiency
        self.model = wrap_model(
            model=self.model,
            grad_sample_mode="ghost",
            batch_first=True,
            loss_reduction="mean",
            max_grad_norm=1.0,
            use_ghost_clipping=True
        )
        
        # Create standard optimizer
        base_optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_params)
        
        # Wrap with DPOptimizer for differential privacy
        self.optimizer = DPOptimizer(
            optimizer=base_optimizer,
            noise_multiplier=1.0,  # Fixed noise multiplier for consistent comparison
            max_grad_norm=1.0,
            expected_batch_size=dataloader.batch_size if dataloader else 1,
        )


def run_local_test():
    """Run local test on CPU with small model (~1M parameters)"""
    print("Starting local CPU test with small model...")
    
    device = "cpu"
    model_config = get_model_config("tiny")  # Very small model for testing
    
    TRAINER_CLASSES = [StandardTrainer, DPSGDTrainer, DPGhostClippingTrainer]
    
    for trainer_cls in TRAINER_CLASSES:
        trainer_name = trainer_cls.__name__
        print(f"\n=== Testing {trainer_name} ===")
        
        # Skip DP trainers if Opacus not available
        if not OPACUS_AVAILABLE and "DP" in trainer_name:
            print(f"[TEST SKIP] {trainer_name} - Opacus not available")
            continue
        
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
            
            # Setup optimizer
            trainer.setup_optimizer(dataloader)
            
            # Run one step (no profiling)
            batch = next(iter(dataloader))
            loss = trainer.train_step(batch)
            
            print(f"[TEST OK] {trainer_name} passed local test. Loss: {loss:.4f}")
            
        except Exception as e:
            print(f"[TEST FAIL] {trainer_name} failed: {str(e)}")
    
    print("\n=== Local testing completed ===")


def run_gpu_profile_step(trainer_name: str, batch_size: int, seq_len: int):
    """
    Core function for NCU profiling on GPU with ~1B parameter model.
    This function will be called by NCU to capture kernel performance.
    """
    print(f"Starting GPU profiling: trainer={trainer_name}, batch_size={batch_size}, seq_len={seq_len}")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! GPU profiling requires CUDA.")
    
    device = "cuda"
    model_config = get_model_config("1b")  # ~1B parameter model
    
    # Create model
    model = SimpleBigModel(**model_config).to(device)
    param_count = model.count_parameters()
    print(f"Model parameters: {param_count:,} ({param_count/1e6:.1f}M)")
    
    # Create dataloader
    dataloader = get_random_dataloader(
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=model_config["vocab_size"],
        device=device,
        num_batches=10
    )
    
    # Setup trainer based on trainer_name
    trainer_classes = {
        "standard": StandardTrainer,
        "dpsgd": DPSGDTrainer,
        "dpsgd_ghost": DPGhostClippingTrainer
    }
    
    if trainer_name not in trainer_classes:
        raise ValueError(f"Unknown trainer: {trainer_name}. Available: {list(trainer_classes.keys())}")
    
    trainer_cls = trainer_classes[trainer_name]
    
    # Skip DP trainers if Opacus not available
    if not OPACUS_AVAILABLE and "DP" in trainer_cls.__name__:
        raise ImportError(f"Opacus not available for {trainer_name}")
    
    # Setup trainer
    optimizer_params = {"lr": 0.001, "momentum": 0.9}
    trainer = trainer_cls(
        model=model,
        optimizer_cls=optim.SGD,
        optimizer_params=optimizer_params,
        device=device
    )
    
    trainer.setup_optimizer(dataloader)
    
    # Get a batch for training
    batch = next(iter(dataloader))
    
    # Warmup steps (NCU will not profile these)
    print("Running warmup steps...")
    for i in range(3):
        loss = trainer.train_step(batch)
        print(f"Warmup step {i+1}, Loss: {loss:.4f}")
    
    # Synchronize before profiled steps
    torch.cuda.synchronize()
    
    # Active profiling steps (NCU will capture these kernels)
    print("Running profiled steps...")
    for i in range(5):
        loss = trainer.train_step(batch)
        print(f"Profiled step {i+1}, Loss: {loss:.4f}")
    
    # Synchronize after profiled steps
    torch.cuda.synchronize()
    
    print("Profiled steps finished.")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="NVIDIA NCU Profiling Script for Opacus DP-SGD")
    parser.add_argument("--mode", type=str, default="profile", 
                       choices=["profile", "test"],
                       help="Mode: 'profile' for GPU profiling with NCU, 'test' for local CPU test")
    parser.add_argument("--trainer", type=str, default="standard",
                       choices=["standard", "dpsgd", "dpsgd_ghost"],
                       help="Trainer type to profile")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for training")
    parser.add_argument("--seq_len", type=int, default=256,
                       help="Sequence length for input")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    if args.mode == "test":
        run_local_test()
    elif args.mode == "profile":
        run_gpu_profile_step(args.trainer, args.batch_size, args.seq_len)
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()


"""
=== NCU 分析指南 (NCU Analysis Guide) ===

1. 如何启动分析:
   在 run_profiling_ncu.sh 运行完毕后，你将得到一堆 .ncu-rep 文件。
   
   查看报告的方法:
   a) 使用 NCU GUI: 
      ncu-ui
      然后在 GUI 中打开 .ncu-rep 文件
   
   b) 使用命令行查看摘要:
      ncu --import report_dpsgd_bs8_seq256.ncu-rep --page details
   
   c) 导出为 CSV 进行批量分析:
      ncu --import report_dpsgd_bs8_seq256.ncu-rep --csv > report_dpsgd_bs8_seq256.csv

2. 如何分析 I/O Cost:
   打开报告后 (例如 report_dpsgd_bs8_seq256.ncu-rep)，在 NCU GUI 中:
   
   a) 主要视图:
      - "Details" 页面: 查看整体性能摘要
      - "GPU Speed of Light" 页面: 查看内存和计算利用率
      - "Memory Workload Analysis" 页面: 详细的内存访问分析
   
   b) 关键指标位置:
      - Memory Throughput 部分: 查看 DRAM 和 L2 缓存吞吐量
      - Compute Workload Analysis: 查看计算与内存的平衡
      - Launch Statistics: 查看 kernel 启动开销

3. (最重要的) 为了证明 per-sample gradient clipping 带来了高昂的 I/O Cost:

   a) 关键指标对比 (dpsgd vs standard):
      - dram_read_throughput: DP-SGD 应该显著更高，因为需要读取更多梯度数据
      - dram_write_throughput: DP-SGD 写入开销更大，存储 per-sample gradients
      - l2_read_throughput / l2_write_throughput: L2 缓存压力增加
      - gld_throughput (Global Load): 全局内存加载吞吐量增加
      - gst_throughput (Global Store): 全局内存存储吞吐量增加
   
   b) 具体分析步骤:
      1) 打开 report_standard_bs8_seq256.ncu-rep 作为基准
      2) 记录关键内存指标的数值
      3) 打开 report_dpsgd_bs8_seq256.ncu-rep 进行对比
      4) 计算百分比增长: (dpsgd_value - standard_value) / standard_value * 100%
   
   c) 预期的性能差异:
      - DRAM 读写吞吐量: DP-SGD 应该比 Standard 高 2-5x
      - L2 缓存吞吐量: 增加 1.5-3x
      - 内存利用率: DP-SGD 接近内存带宽上限
      - Kernel 执行时间: DP-SGD 中梯度相关 kernel 时间显著增加
   
   d) Ghost Clipping 的优化效果:
      - report_dpsgd_ghost_bs8_seq256.ncu-rep 应该显示:
      - 相比 flat clipping 减少的内存吞吐量需求
      - 更好的内存访问模式
      - 但仍然高于 standard SGD 的基准

   e) 关注的 Kernel 类型:
      - 包含 "grad", "clip", "norm" 的 kernel 名称
      - "elementwise" 操作 (梯度裁剪中的逐元素计算)
      - "reduce" 操作 (计算梯度范数)
      - "copy" 或 "memcpy" 操作 (梯度数据传输)

4. 结论导出:
   通过对比这些指标，你可以量化证明:
   - Per-sample gradient clipping 确实带来了显著的 I/O 开销
   - Ghost Clipping 等优化技术的内存效率改进
   - 不同 batch size 和 sequence length 对内存压力的影响

   建议制作表格对比各个配置下的关键指标，以便清晰展示性能差异。
"""