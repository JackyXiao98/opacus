#!/usr/bin/env python3
"""
GradSampleModule vs GradSampleModuleFastGradientClipping 详细对比

这个示例展示了两种不同的per-sample gradient计算方法的区别：
1. 标准版本：计算完整的per-sample gradients
2. 快速版本：只计算梯度范数，支持Ghost Clipping和Fast Gradient Clipping
"""

import torch
import torch.nn as nn
import tracemalloc
from opacus.grad_sample import GradSampleModule

def create_test_model():
    """创建测试模型"""
    return nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )

def create_test_data(batch_size=32, input_dim=100):
    """创建测试数据"""
    X = torch.randn(batch_size, input_dim)
    y = torch.randn(batch_size, 1)
    return X, y

def compare_basic_functionality():
    """对比基本功能差异"""
    print("=" * 80)
    print("1. 基本功能对比")
    print("=" * 80)
    
    # 创建两个相同的模型
    model1 = create_test_model()
    model2 = create_test_model()
    
    # 确保参数相同
    with torch.no_grad():
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            p2.copy_(p1)
    
    # 包装成不同的GradSampleModule
    grad_sample_module = GradSampleModule(model1)
    
    # 注意：这里我们无法直接导入GradSampleModuleFastGradientClipping
    # 因为它可能在不同的文件中，所以我们用概念性的对比
    
    print("标准GradSampleModule特点:")
    print("- 计算完整的per-sample gradients")
    print("- 存储在参数的grad_sample属性中")
    print("- 形状: [batch_size, *param_shape]")
    print("- 内存使用: 高（存储所有梯度）")
    
    print("\nFastGradientClipping版本特点:")
    print("- 只计算梯度范数")
    print("- 存储在参数的_norm_sample属性中")
    print("- 形状: [batch_size] (只有范数)")
    print("- 内存使用: 低（只存储范数）")
    
    # 测试标准版本
    X, y = create_test_data(batch_size=4)
    output = grad_sample_module(X)
    loss = nn.MSELoss()(output, y)
    loss.backward()
    
    print(f"\n标准版本结果:")
    for name, param in grad_sample_module.named_parameters():
        if hasattr(param, 'grad_sample'):
            print(f"  {name}: grad_sample形状 = {param.grad_sample.shape}")
            # 计算范数用于对比
            norms = param.grad_sample.reshape(param.grad_sample.shape[0], -1).norm(2, dim=1)
            print(f"    计算得到的范数: {norms}")

def explain_memory_efficiency():
    """解释内存效率差异"""
    print(f"\n" + "=" * 80)
    print("2. 内存效率对比")
    print("=" * 80)
    
    batch_size = 32
    param_sizes = {
        "layer1.weight": (50, 100),    # 5000 parameters
        "layer1.bias": (50,),          # 50 parameters  
        "layer2.weight": (10, 50),     # 500 parameters
        "layer2.bias": (10,),          # 10 parameters
        "layer3.weight": (1, 10),      # 10 parameters
        "layer3.bias": (1,),           # 1 parameter
    }
    
    print("内存使用估算 (假设float32, 4 bytes per element):")
    print(f"Batch size: {batch_size}")
    
    total_standard = 0
    total_fast = 0
    
    for name, shape in param_sizes.items():
        param_count = torch.tensor(shape).prod().item()
        
        # 标准版本：存储完整梯度 [batch_size, *param_shape]
        standard_memory = batch_size * param_count * 4  # bytes
        
        # 快速版本：只存储范数 [batch_size]
        fast_memory = batch_size * 4  # bytes
        
        total_standard += standard_memory
        total_fast += fast_memory
        
        print(f"\n{name}:")
        print(f"  参数形状: {shape} ({param_count} 参数)")
        print(f"  标准版本: {batch_size} × {param_count} × 4 = {standard_memory:,} bytes")
        print(f"  快速版本: {batch_size} × 1 × 4 = {fast_memory:,} bytes")
        print(f"  节省: {(1 - fast_memory/standard_memory)*100:.1f}%")
    
    print(f"\n总计:")
    print(f"  标准版本总内存: {total_standard:,} bytes ({total_standard/1024/1024:.2f} MB)")
    print(f"  快速版本总内存: {total_fast:,} bytes ({total_fast/1024:.2f} KB)")
    print(f"  总体节省: {(1 - total_fast/total_standard)*100:.1f}%")

def explain_ghost_clipping():
    """解释Ghost Clipping机制"""
    print(f"\n" + "=" * 80)
    print("3. Ghost Clipping vs Fast Gradient Clipping")
    print("=" * 80)
    
    explanation = """
    FastGradientClipping版本支持两种优化模式:
    
    🔥 Ghost Clipping (use_ghost_clipping=True):
    ----------------------------------------
    • 原理: 直接从激活值和反向梯度计算范数，无需计算完整梯度
    • 支持层: 有专门NORM_SAMPLERS的层（如Linear, Conv2d等）
    • 内存效率: 最高（完全避免梯度实例化）
    • 计算效率: 最高（专门优化的范数计算）
    • 限制: 不支持参数共享(parameter tying)
    
    ⚡ Fast Gradient Clipping (use_ghost_clipping=False):
    --------------------------------------------------
    • 原理: 先计算完整梯度，然后立即计算范数并丢弃梯度
    • 支持层: 所有层（使用GRAD_SAMPLERS或functorch）
    • 内存效率: 中等（临时存储梯度）
    • 计算效率: 中等（需要完整梯度计算）
    • 限制: 较少
    
    📊 标准方法 (GradSampleModule):
    -----------------------------
    • 原理: 计算并持久存储完整的per-sample梯度
    • 支持层: 所有层
    • 内存效率: 最低（存储所有梯度）
    • 计算效率: 最低（需要存储大量数据）
    • 限制: 无（最通用）
    """
    print(explanation)

def demonstrate_workflow_differences():
    """演示工作流程差异"""
    print(f"\n" + "=" * 80)
    print("4. 工作流程差异")
    print("=" * 80)
    
    print("标准GradSampleModule工作流程:")
    print("1. 前向传播 → 保存激活值")
    print("2. 反向传播 → 计算per-sample梯度")
    print("3. 存储梯度 → param.grad_sample = [batch_size, *param_shape]")
    print("4. 优化器使用 → 从grad_sample计算范数、裁剪、聚合")
    
    print(f"\nFastGradientClipping工作流程:")
    print("Ghost Clipping模式:")
    print("1. 前向传播 → 保存激活值")
    print("2. 反向传播 → 直接计算梯度范数（无梯度实例化）")
    print("3. 存储范数 → param._norm_sample = [batch_size]")
    print("4. 优化器使用 → 直接使用范数进行裁剪")
    
    print(f"\nFast Gradient Clipping模式:")
    print("1. 前向传播 → 保存激活值")
    print("2. 反向传播 → 计算per-sample梯度")
    print("3. 计算范数 → 立即从梯度计算范数")
    print("4. 丢弃梯度 → param.grad_sample = None")
    print("5. 存储范数 → param._norm_sample = [batch_size]")
    print("6. 优化器使用 → 使用范数进行裁剪")

def compare_use_cases():
    """对比使用场景"""
    print(f"\n" + "=" * 80)
    print("5. 适用场景对比")
    print("=" * 80)
    
    scenarios = {
        "标准GradSampleModule": {
            "适用场景": [
                "需要完整per-sample梯度信息",
                "研究和调试目的",
                "自定义梯度处理逻辑",
                "小模型或内存充足的情况"
            ],
            "优势": [
                "功能最完整",
                "最大灵活性",
                "支持所有操作"
            ],
            "劣势": [
                "内存使用量大",
                "计算开销高",
                "可能导致OOM"
            ]
        },
        "FastGradientClipping": {
            "适用场景": [
                "只需要梯度裁剪功能",
                "大模型训练",
                "内存受限环境",
                "生产环境部署"
            ],
            "优势": [
                "内存效率极高",
                "计算速度快",
                "支持Ghost Clipping优化"
            ],
            "劣势": [
                "功能相对受限",
                "不支持参数共享(Ghost模式)",
                "调试信息较少"
            ]
        }
    }
    
    for method, info in scenarios.items():
        print(f"\n{method}:")
        print(f"  适用场景:")
        for scenario in info["适用场景"]:
            print(f"    • {scenario}")
        print(f"  优势:")
        for advantage in info["优势"]:
            print(f"    ✅ {advantage}")
        print(f"  劣势:")
        for disadvantage in info["劣势"]:
            print(f"    ❌ {disadvantage}")

def explain_implementation_differences():
    """解释实现差异"""
    print(f"\n" + "=" * 80)
    print("6. 关键实现差异")
    print("=" * 80)
    
    print("capture_backprops_hook方法差异:")
    print(f"\n标准版本:")
    print("```python")
    print("# 总是计算完整梯度")
    print("grad_samples = grad_sampler_fn(module, activations, backprops)")
    print("for param, gs in grad_samples.items():")
    print("    create_or_accumulate_grad_sample(param=param, grad_sample=gs)")
    print("```")
    
    print(f"\n快速版本:")
    print("```python")
    print("if self.use_ghost_clipping and type(module) in self.NORM_SAMPLERS:")
    print("    # Ghost Clipping: 直接计算范数")
    print("    norm_sampler_fn = self.NORM_SAMPLERS[type(module)]")
    print("    norm_samples = norm_sampler_fn(module, activations, backprops)")
    print("    for param, ns in norm_samples.items():")
    print("        param._norm_sample = ns")
    print("else:")
    print("    # Fast Gradient Clipping: 计算梯度后立即转换为范数")
    print("    grad_samples = grad_sampler_fn(module, activations, backprops)")
    print("    # ... 计算范数并丢弃梯度")
    print("    create_norm_sample(param=p, grad_sample=p.grad_sample)")
    print("    p.grad_sample = None  # 立即释放内存")
    print("```")
    
    print(f"\n新增方法:")
    print("• get_norm_sample(): 获取per-sample梯度范数")
    print("• get_clipping_coef(): 计算裁剪系数")
    print("• NORM_SAMPLERS: 专门的范数计算器注册表")

if __name__ == "__main__":
    print("GradSampleModule vs GradSampleModuleFastGradientClipping 详细对比")
    print("=" * 80)
    
    compare_basic_functionality()
    explain_memory_efficiency()
    explain_ghost_clipping()
    demonstrate_workflow_differences()
    compare_use_cases()
    explain_implementation_differences()
    
    print(f"\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("选择建议:")
    print("• 如果只需要梯度裁剪 → 使用 FastGradientClipping")
    print("• 如果需要完整梯度信息 → 使用标准 GradSampleModule") 
    print("• 如果内存受限 → 优先考虑 FastGradientClipping + Ghost Clipping")
    print("• 如果有参数共享 → 使用 FastGradientClipping (非Ghost模式)")
    print("=" * 80)