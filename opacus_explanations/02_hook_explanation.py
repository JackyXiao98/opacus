#!/usr/bin/env python3
"""
capture_backprops_hook 函数工作原理演示

这个示例展示了hook函数如何在反向传播过程中被调用，
以及它如何计算和存储per-sample gradients。
"""

import torch
import torch.nn as nn
from opacus.grad_sample import GradSampleModule

def demonstrate_hook_mechanism():
    """演示hook机制的工作原理"""
    print("=" * 60)
    print("capture_backprops_hook 工作原理演示")
    print("=" * 60)
    
    # 创建一个简单的线性层
    linear_layer = nn.Linear(3, 2)
    print(f"原始线性层参数形状:")
    print(f"  weight: {linear_layer.weight.shape}")
    print(f"  bias: {linear_layer.bias.shape}")
    
    # 包装成GradSampleModule
    grad_sample_module = GradSampleModule(linear_layer)
    
    # 创建batch数据 (batch_size=4, input_dim=3)
    batch_size = 4
    input_data = torch.randn(batch_size, 3)
    target = torch.randn(batch_size, 2)
    
    print(f"\n输入数据形状: {input_data.shape}")
    print(f"目标数据形状: {target.shape}")
    
    # 前向传播
    output = grad_sample_module(input_data)
    loss = nn.MSELoss(reduction='none')(output, target)  # 注意：reduction='none'
    
    print(f"\n输出形状: {output.shape}")
    print(f"损失形状: {loss.shape}")
    
    # 反向传播 - 这时会触发capture_backprops_hook
    print(f"\n开始反向传播...")
    loss.sum().backward()
    
    # 检查生成的per-sample gradients
    print(f"\n反向传播后的per-sample gradients:")
    for name, param in grad_sample_module.named_parameters():
        if hasattr(param, 'grad_sample'):
            print(f"  {name}:")
            print(f"    原始参数形状: {param.shape}")
            print(f"    grad_sample形状: {param.grad_sample.shape}")
            print(f"    第一维是batch_size: {param.grad_sample.shape[0] == batch_size}")

def explain_hook_process():
    """解释hook处理过程的详细步骤"""
    print(f"\n" + "=" * 60)
    print("Hook处理过程详解")
    print("=" * 60)
    
    explanation = """
    capture_backprops_hook 的处理过程：
    
    1. **Hook触发时机**:
       - 在每个模块的反向传播完成时自动调用
       - PyTorch的hook机制确保在正确的时间点执行
    
    2. **获取反向传播信息**:
       - forward_output[0] 包含了从后续层传回的梯度
       - 这些梯度对应于当前层输出的梯度
    
    3. **激活值的使用**:
       - 前向传播时，capture_activations_hook 已经保存了激活值
       - 反向传播时，结合激活值和反向梯度计算参数梯度
    
    4. **Per-sample梯度计算**:
       - 对于线性层: grad_weight = einsum('ni,nj->nij', backprops, activations)
       - 对于偏置: grad_bias = backprops
       - 'n' 维度对应batch中的每个样本
    
    5. **梯度存储**:
       - 将计算出的梯度存储在参数的 grad_sample 属性中
       - 形状为 [batch_size, *param_shape]
    
    6. **特殊情况处理**:
       - RNN: 需要累积多个时间步的梯度
       - 参数共享: 需要处理同一参数被多次使用的情况
       - 梯度累积: 控制是否允许跨batch累积梯度
    """
    print(explanation)

def demonstrate_gradient_computation():
    """演示具体的梯度计算过程"""
    print(f"\n" + "=" * 60)
    print("具体梯度计算演示")
    print("=" * 60)
    
    # 手动模拟hook中的梯度计算过程
    batch_size = 3
    input_dim = 2
    output_dim = 2
    
    # 模拟激活值（前向传播的输入）
    activations = torch.randn(batch_size, input_dim)
    print(f"激活值 (输入) 形状: {activations.shape}")
    print(f"激活值内容:\n{activations}")
    
    # 模拟反向传播梯度（从后续层传回的梯度）
    backprops = torch.randn(batch_size, output_dim)
    print(f"\n反向传播梯度形状: {backprops.shape}")
    print(f"反向传播梯度内容:\n{backprops}")
    
    # 计算权重的per-sample梯度（模拟线性层的梯度计算）
    weight_grad_samples = torch.einsum('ni,nj->nij', backprops, activations)
    print(f"\n权重的per-sample梯度形状: {weight_grad_samples.shape}")
    print(f"解释: [batch_size={batch_size}, output_dim={output_dim}, input_dim={input_dim}]")
    
    # 计算偏置的per-sample梯度
    bias_grad_samples = backprops
    print(f"\n偏置的per-sample梯度形状: {bias_grad_samples.shape}")
    print(f"解释: [batch_size={batch_size}, output_dim={output_dim}]")
    
    # 展示每个样本的梯度
    print(f"\n每个样本的权重梯度:")
    for i in range(batch_size):
        print(f"  样本 {i}: {weight_grad_samples[i].shape}")
        print(f"    范数: {weight_grad_samples[i].norm().item():.4f}")

if __name__ == "__main__":
    demonstrate_hook_mechanism()
    explain_hook_process()
    demonstrate_gradient_computation()
    
    print(f"\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)