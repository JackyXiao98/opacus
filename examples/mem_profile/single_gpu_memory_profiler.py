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
import copy
import logging
import os
import random
from typing import Dict, Any, Optional

import torch
from opacus.grad_sample import GradSampleModule
from opacus.optimizers import DPOptimizer
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPUMemoryProfiler:
    """GPU内存分析器，用于跟踪和分析内存使用情况"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.is_cuda = device == "cuda" and torch.cuda.is_available()
        self.memory_stats = []
        
    def reset_stats(self):
        """重置内存统计"""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        self.memory_stats = []
    
    def get_memory_stats(self) -> Dict[str, float]:
        """获取当前内存统计信息（以MB为单位）"""
        if not self.is_cuda:
            return {"allocated": 0, "reserved": 0, "max_allocated": 0}
        
        return {
            "allocated": torch.cuda.memory_allocated(self.device) / 2**20,
            "reserved": torch.cuda.memory_reserved(self.device) / 2**20,
            "max_allocated": torch.cuda.max_memory_allocated(self.device) / 2**20,
        }
    
    def profile_function(self, func, *args, stage_name: str = "operation", **kwargs):
        """分析函数执行时的内存使用情况"""
        # 记录执行前的内存状态
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        mem_before = self.get_memory_stats()
        
        # 执行函数
        result = func(*args, **kwargs)
        
        # 记录执行后的内存状态
        mem_after = self.get_memory_stats()
        
        # 计算内存差异
        mem_diff = {
            "allocated_diff": mem_after["allocated"] - mem_before["allocated"],
            "reserved_diff": mem_after["reserved"] - mem_before["reserved"],
            "peak_allocated": mem_after["max_allocated"],
        }
        
        # 记录统计信息
        stats = {
            "stage": stage_name,
            "before": mem_before,
            "after": mem_after,
            "diff": mem_diff,
        }
        self.memory_stats.append(stats)
        
        # 打印内存使用情况
        if self.is_cuda:
            print(f"\n=== {stage_name} Memory Usage ===")
            print(f"Before: {mem_before['allocated']:.2f} MB allocated, {mem_before['reserved']:.2f} MB reserved")
            print(f"After:  {mem_after['allocated']:.2f} MB allocated, {mem_after['reserved']:.2f} MB reserved")
            print(f"Peak:   {mem_after['max_allocated']:.2f} MB allocated")
            print(f"Diff:   {mem_diff['allocated_diff']:+.2f} MB allocated, {mem_diff['reserved_diff']:+.2f} MB reserved")
        else:
            print(f"\n=== {stage_name} Completed ===")
            print("(CPU模式 - 内存统计不可用)")
        
        return result
    
    def print_summary(self):
        """打印内存使用总结"""
        print("\n" + "="*60)
        if self.is_cuda:
            print("MEMORY USAGE SUMMARY")
        else:
            print("EXECUTION SUMMARY (CPU模式)")
        print("="*60)
        
        for i, stats in enumerate(self.memory_stats):
            print(f"{i+1}. {stats['stage']}")
            if self.is_cuda:
                print(f"   Peak: {stats['diff']['peak_allocated']:.2f} MB")
                print(f"   Diff: {stats['diff']['allocated_diff']:+.2f} MB")
            else:
                print("   已完成 (CPU模式)")


def get_model_config(model_size: str = "small") -> LlamaConfig:
    """获取不同大小的模型配置"""
    configs = {
        "tiny": LlamaConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
        ),
        "small": LlamaConfig(
            vocab_size=32000,
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            max_position_embeddings=2048,
        ),
        "medium": LlamaConfig(
            vocab_size=32000,
            hidden_size=1024,
            intermediate_size=2048,
            num_hidden_layers=8,
            num_attention_heads=16,
            num_key_value_heads=8,
            max_position_embeddings=2048,
        ),
        "1b": LlamaConfig(
            vocab_size=32000,
            hidden_size=2048,
            intermediate_size=5504,
            num_hidden_layers=22,
            num_attention_heads=32,
            num_key_value_heads=4,
            max_position_embeddings=4096,
        ),
        "3b": LlamaConfig(
            vocab_size=32000,
            hidden_size=3200,
            intermediate_size=8640,
            num_hidden_layers=26,
            num_attention_heads=32,
            num_key_value_heads=32,
            max_position_embeddings=4096,
        ),
    }
    return configs.get(model_size, configs["small"])


def generate_sample_data(config: LlamaConfig, batch_size: int = 1, seq_length: int = 128):
    """生成样本数据"""
    input_ids = torch.randint(0, config.vocab_size, size=(batch_size, seq_length))
    labels = torch.randint(0, config.vocab_size, size=(batch_size, seq_length))
    return input_ids, labels


def create_model(config: LlamaConfig, device: str = "cuda") -> LlamaForCausalLM:
    """创建并初始化模型"""
    model = LlamaForCausalLM(config).to(device)
    model.train()
    
    # 设置随机种子以确保可重现性
    random.seed(42)
    torch.manual_seed(42)
    model.init_weights()
    
    return model


def setup_dp_training(model, learning_rate: float = 1e-4, noise_multiplier: float = 1.0, 
                     max_grad_norm: float = 1.0, batch_size: int = 1):
    """设置差分隐私训练"""
    # 包装模型以支持差分隐私
    dp_model = GradSampleModule(model)
    
    # 创建优化器
    optimizer = torch.optim.SGD(dp_model.parameters(), lr=learning_rate)
    dp_optimizer = DPOptimizer(
        optimizer,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        expected_batch_size=batch_size,
    )
    
    # 创建损失函数
    criterion = torch.nn.CrossEntropyLoss()
    
    return dp_model, dp_optimizer, criterion


def run_training_step(model, optimizer, criterion, input_ids, labels, profiler: GPUMemoryProfiler):
    """执行一个训练步骤并分析内存使用"""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    
    # 前向传播
    def forward_pass():
        return model(input_ids, labels=labels)
    
    output = profiler.profile_function(forward_pass, stage_name="Forward Pass")
    
    # 获取损失（LlamaForCausalLM在给定labels时会自动计算损失）
    loss = output.loss
    
    # 反向传播
    def backward_pass():
        loss.backward()
    
    profiler.profile_function(backward_pass, stage_name="Backward Pass")
    
    # 优化器步骤
    def optimizer_step():
        optimizer.step()
    
    profiler.profile_function(optimizer_step, stage_name="Optimizer Step")
    
    # 清零梯度
    def zero_grad():
        optimizer.zero_grad()
    
    profiler.profile_function(zero_grad, stage_name="Zero Gradients")
    
    return loss.item()


def run_memory_analysis(model_size: str = "3b", batch_size: int = 1, seq_length: int = 128, 
                       device: str = "cuda", verbose: bool = False):
    """运行内存分析并返回结果数据"""
    # 检查CUDA可用性
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    if verbose:
        print(f"使用设备: {device}")
        print(f"模型大小: {model_size}")
        print(f"批次大小: {batch_size}")
        print(f"序列长度: {seq_length}")
    
    # 创建内存分析器
    profiler = GPUMemoryProfiler(device)
    
    # 获取模型配置
    config = get_model_config(model_size)
    
    # 创建模型
    def create_model_func():
        return create_model(config, device)
    
    model = profiler.profile_function(create_model_func, stage_name="Model Creation")
    
    # 设置差分隐私训练
    def setup_dp_func():
        return setup_dp_training(
            model, 
            learning_rate=1e-4,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            batch_size=batch_size
        )
    
    dp_model, dp_optimizer, dp_criterion = profiler.profile_function(
        setup_dp_func, stage_name="DP Setup"
    )
    
    # 生成样本数据
    input_ids, labels = generate_sample_data(config, batch_size, seq_length)
    
    # 执行一个训练步骤
    loss = run_training_step(dp_model, dp_optimizer, dp_criterion, input_ids, labels, profiler)
    
    # 返回内存统计数据
    return profiler.memory_stats


def main():
    parser = argparse.ArgumentParser(description="单GPU LLaMA内存分析工具")
    parser.add_argument("--model_size", choices=["tiny", "small", "medium", "1b", "3b"], default="small",
                       help="模型大小")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--seq_length", type=int, default=128, help="序列长度")
    parser.add_argument("--num_steps", type=int, default=3, help="训练步数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--noise_multiplier", type=float, default=1.0, help="噪声倍数")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="最大梯度范数")
    parser.add_argument("--device", default="cuda", help="设备")
    
    args = parser.parse_args()
    
    # 检查CUDA可用性
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA不可用，切换到CPU")
        args.device = "cpu"
    
    print(f"使用设备: {args.device}")
    print(f"模型大小: {args.model_size}")
    print(f"批次大小: {args.batch_size}")
    print(f"序列长度: {args.seq_length}")
    print(f"训练步数: {args.num_steps}")
    
    # 创建内存分析器
    profiler = GPUMemoryProfiler(args.device)
    
    # 获取模型配置
    config = get_model_config(args.model_size)
    print(f"\n模型配置:")
    print(f"  词汇表大小: {config.vocab_size}")
    print(f"  隐藏层大小: {config.hidden_size}")
    print(f"  层数: {config.num_hidden_layers}")
    print(f"  注意力头数: {config.num_attention_heads}")
    
    # 创建模型
    def create_model_func():
        return create_model(config, args.device)
    
    model = profiler.profile_function(create_model_func, stage_name="Model Creation")
    
    # 设置差分隐私训练
    def setup_dp_func():
        return setup_dp_training(
            model, 
            learning_rate=args.learning_rate,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
            batch_size=args.batch_size
        )
    
    dp_model, dp_optimizer, dp_criterion = profiler.profile_function(
        setup_dp_func, stage_name="DP Setup"
    )
    
    # 生成样本数据
    input_ids, labels = generate_sample_data(config, args.batch_size, args.seq_length)
    
    print(f"\n开始训练 {args.num_steps} 步...")
    
    # 执行训练步骤
    for step in range(args.num_steps):
        print(f"\n--- 训练步骤 {step + 1}/{args.num_steps} ---")
        loss = run_training_step(dp_model, dp_optimizer, dp_criterion, input_ids, labels, profiler)
        print(f"损失: {loss:.4f}")
    
    # 打印总结
    profiler.print_summary()
    
    # 打印最终内存状态
    final_stats = profiler.get_memory_stats()
    print(f"\n最终内存状态:")
    print(f"  已分配: {final_stats['allocated']:.2f} MB")
    print(f"  已保留: {final_stats['reserved']:.2f} MB")
    print(f"  峰值分配: {final_stats['max_allocated']:.2f} MB")


if __name__ == "__main__":
    main()