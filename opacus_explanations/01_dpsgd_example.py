#!/usr/bin/env python3
"""
DPSGD (Differentially Private Stochastic Gradient Descent) 最小可行示例

这个示例展示了Opacus如何实现DPSGD的核心机制：
1. 计算per-sample gradients
2. 计算每个样本的梯度范数
3. 梯度裁剪
4. 添加高斯噪声
"""

import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.grad_sample import GradSampleModule
from opacus.optimizers import DPOptimizer

# 设置随机种子以便复现
torch.manual_seed(42)

def create_simple_model():
    """创建一个简单的线性模型"""
    return nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )

def create_synthetic_data(batch_size=4, input_dim=10):
    """创建合成数据"""
    X = torch.randn(batch_size, input_dim)
    y = torch.randn(batch_size, 1)
    return X, y

def demonstrate_per_sample_gradients():
    """演示per-sample gradient的计算"""
    print("=" * 60)
    print("1. Per-Sample Gradient 计算演示")
    print("=" * 60)
    
    # 创建模型和数据
    model = create_simple_model()
    X, y = create_synthetic_data(batch_size=3)
    
    # 包装模型以计算per-sample gradients
    grad_sample_module = GradSampleModule(model)
    
    # 前向传播
    output = grad_sample_module(X)
    loss = nn.MSELoss(reduction='none')(output, y)  # 注意：reduction='none'
    
    # 反向传播
    loss.sum().backward()
    
    # 检查per-sample gradients
    for name, param in grad_sample_module.named_parameters():
        if hasattr(param, 'grad_sample'):
            print(f"\n参数: {name}")
            print(f"参数形状: {param.shape}")
            print(f"Per-sample gradient形状: {param.grad_sample.shape}")
            print(f"Per-sample gradient前3个样本的范数:")
            for i in range(3):
                norm = param.grad_sample[i].norm().item()
                print(f"  样本 {i}: {norm:.4f}")

def demonstrate_gradient_clipping():
    """演示梯度裁剪机制"""
    print("\n" + "=" * 60)
    print("2. 梯度裁剪演示")
    print("=" * 60)
    
    # 创建模型和数据
    model = create_simple_model()
    X, y = create_synthetic_data(batch_size=4)
    
    # 包装模型
    grad_sample_module = GradSampleModule(model)
    
    # 创建优化器
    optimizer = optim.SGD(grad_sample_module.parameters(), lr=0.01)
    dp_optimizer = DPOptimizer(
        optimizer=optimizer,
        noise_multiplier=1.0,
        max_grad_norm=1.0,  # 最大梯度范数
        expected_batch_size=4,
    )
    
    # 前向传播
    output = grad_sample_module(X)
    loss = nn.MSELoss()(output, y)
    
    # 反向传播
    loss.backward()
    
    # 手动演示裁剪过程
    print("裁剪前的per-sample gradient范数:")
    per_sample_norms = []
    for param in grad_sample_module.parameters():
        if hasattr(param, 'grad_sample'):
            # 计算每个样本的梯度范数
            grad_sample_flat = param.grad_sample.reshape(len(param.grad_sample), -1)
            param_norms = grad_sample_flat.norm(2, dim=-1)
            per_sample_norms.append(param_norms)
    
    # 计算总的per-sample范数
    if per_sample_norms:
        total_norms = torch.stack(per_sample_norms, dim=1).norm(2, dim=1)
        print(f"每个样本的梯度范数: {total_norms}")
        
        # 计算裁剪因子
        max_grad_norm = 1.0
        clip_factors = (max_grad_norm / (total_norms + 1e-6)).clamp(max=1.0)
        print(f"裁剪因子: {clip_factors}")
        print(f"裁剪后的范数: {total_norms * clip_factors}")

def demonstrate_noise_addition():
    """演示噪声添加"""
    print("\n" + "=" * 60)
    print("3. 噪声添加演示")
    print("=" * 60)
    
    # 模拟裁剪后的梯度
    clipped_grad = torch.tensor([0.5, -0.3, 0.8, 0.1])
    
    # DPSGD参数
    noise_multiplier = 1.0
    max_grad_norm = 1.0
    
    # 计算噪声标准差
    noise_std = noise_multiplier * max_grad_norm
    print(f"噪声标准差: {noise_std}")
    
    # 添加高斯噪声
    noise = torch.normal(0, noise_std, size=clipped_grad.shape)
    noisy_grad = clipped_grad + noise
    
    print(f"原始梯度: {clipped_grad}")
    print(f"噪声: {noise}")
    print(f"加噪后梯度: {noisy_grad}")

def complete_dpsgd_example():
    """完整的DPSGD训练示例"""
    print("\n" + "=" * 60)
    print("4. 完整DPSGD训练示例")
    print("=" * 60)
    
    # 创建模型和数据
    model = create_simple_model()
    X, y = create_synthetic_data(batch_size=8)
    
    # 创建正确的数据加载器
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # 使用PrivacyEngine自动设置
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    privacy_engine = PrivacyEngine()
    
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
    )
    
    print("训练前的参数:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.data.flatten()[:3]}...")
    
    # 训练一步
    model.train()
    for batch_X, batch_y in data_loader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = nn.MSELoss()(output, batch_y)
        loss.backward()
        optimizer.step()
        break
    
    print("\n训练后的参数:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.data.flatten()[:3]}...")
    
    print(f"\n损失: {loss.item():.4f}")

def explain_dpsgd_algorithm():
    """解释DPSGD算法的核心步骤"""
    print("\n" + "=" * 60)
    print("DPSGD算法核心步骤解释")
    print("=" * 60)
    
    algorithm_steps = """
    DPSGD算法的核心步骤：
    
    1. **Per-Sample Gradient计算**:
       - 对于batch中的每个样本，单独计算梯度
       - 使用hooks机制在反向传播时捕获每层的激活值和梯度
       - 存储在参数的grad_sample属性中
    
    2. **梯度范数计算**:
       - 计算每个样本所有参数梯度的L2范数
       - per_sample_norms = ||∇θ L(θ, xi)||₂
    
    3. **梯度裁剪**:
       - 计算裁剪因子: clip_factor = min(1, C / ||∇θ L(θ, xi)||₂)
       - 裁剪梯度: ∇̃θ L(θ, xi) = clip_factor × ∇θ L(θ, xi)
       - 其中C是最大梯度范数阈值
    
    4. **梯度聚合**:
       - 将裁剪后的per-sample梯度求和
       - ∇̃θ = (1/B) × Σᵢ ∇̃θ L(θ, xi)
    
    5. **噪声添加**:
       - 添加高斯噪声: ∇̃θ = ∇̃θ + N(0, σ²I)
       - 其中σ = noise_multiplier × C
    
    6. **参数更新**:
       - 使用加噪后的梯度更新参数
       - θ = θ - η × ∇̃θ
    """
    print(algorithm_steps)

if __name__ == "__main__":
    print("Opacus DPSGD 实现原理演示")
    print("=" * 60)
    
    # 运行所有演示
    demonstrate_per_sample_gradients()
    demonstrate_gradient_clipping()
    demonstrate_noise_addition()
    complete_dpsgd_example()
    explain_dpsgd_algorithm()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)