#!/usr/bin/env python3
"""
ft_compute_sample_grad 函数实现原理演示

这个示例展示了Opacus如何使用functorch动态创建per-sample gradient计算函数
"""

import torch
import torch.nn as nn
from torch.func import grad, vmap
import copy
from contextlib import nullcontext

def demonstrate_functorch_mechanism():
    """演示functorch机制如何工作"""
    print("=" * 60)
    print("ft_compute_sample_grad 动态创建过程演示")
    print("=" * 60)
    
    # 创建一个简单的线性层
    layer = nn.Linear(3, 2)
    print(f"原始层: {layer}")
    print(f"参数: weight {layer.weight.shape}, bias {layer.bias.shape}")
    
    # 检查是否已有ft_compute_sample_grad方法
    print(f"\n初始状态 - 是否有ft_compute_sample_grad: {hasattr(layer, 'ft_compute_sample_grad')}")
    
    # 手动执行prepare_layer的核心逻辑
    print(f"\n开始prepare_layer过程...")
    
    # 1. 创建functional版本的模型
    flayer, params = make_functional_demo(layer)
    print(f"创建functional模型完成")
    
    # 2. 定义损失计算函数
    def compute_loss_stateless_model(params, activations, backprops):
        # 为单个样本添加batch维度
        batched_activations = activations.unsqueeze(0)
        batched_backprops = backprops.unsqueeze(0)
        
        # 前向传播
        output = flayer(params, batched_activations)
        # 计算"损失"（实际上是梯度传播）
        loss = (output * batched_backprops).sum()
        return loss
    
    print(f"定义损失计算函数完成")
    
    # 3. 创建梯度计算函数
    ft_compute_grad = grad(compute_loss_stateless_model)
    print(f"创建梯度计算函数完成")
    
    # 4. 使用vmap创建批量梯度计算函数
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0, 0))
    print(f"创建批量梯度计算函数完成")
    
    # 5. 绑定到layer对象
    layer.ft_compute_sample_grad = ft_compute_sample_grad
    print(f"绑定到layer对象完成")
    
    print(f"\n最终状态 - 是否有ft_compute_sample_grad: {hasattr(layer, 'ft_compute_sample_grad')}")
    
    return layer

def make_functional_demo(mod: nn.Module):
    """简化版的make_functional函数"""
    params_dict = dict(mod.named_parameters())
    params_names = params_dict.keys()
    params_values = tuple(params_dict.values())
    
    stateless_mod = copy.deepcopy(mod)
    stateless_mod.to("meta")
    
    def fmodel(new_params_values, *args, **kwargs):
        new_params_dict = {
            name: value for name, value in zip(params_names, new_params_values)
        }
        return torch.func.functional_call(stateless_mod, new_params_dict, args, kwargs)
    
    return fmodel, params_values

def test_ft_compute_sample_grad():
    """测试动态创建的ft_compute_sample_grad函数"""
    print(f"\n" + "=" * 60)
    print("测试ft_compute_sample_grad函数")
    print("=" * 60)
    
    # 获取准备好的layer
    layer = demonstrate_functorch_mechanism()
    
    # 创建测试数据
    batch_size = 3
    activations = torch.randn(batch_size, 3)  # [batch_size, input_dim]
    backprops = torch.randn(batch_size, 2)    # [batch_size, output_dim]
    
    print(f"\n测试数据:")
    print(f"activations形状: {activations.shape}")
    print(f"backprops形状: {backprops.shape}")
    
    # 获取参数列表
    parameters = list(layer.parameters())
    print(f"参数数量: {len(parameters)}")
    
    # 调用ft_compute_sample_grad
    print(f"\n调用ft_compute_sample_grad...")
    per_sample_grads = layer.ft_compute_sample_grad(parameters, activations, backprops)
    
    print(f"返回的per-sample梯度:")
    for i, grad in enumerate(per_sample_grads):
        param_name = "weight" if i == 0 else "bias"
        print(f"  {param_name} per-sample梯度形状: {grad.shape}")
        print(f"  解释: [batch_size={batch_size}, *param_shape]")

def explain_vmap_mechanism():
    """解释vmap机制"""
    print(f"\n" + "=" * 60)
    print("vmap机制解释")
    print("=" * 60)
    
    explanation = """
    vmap (vectorized map) 的工作原理:
    
    1. **基础函数**: ft_compute_grad(params, single_activation, single_backprop)
       - 输入: 参数 + 单个样本的激活值 + 单个样本的反向梯度
       - 输出: 该样本对应的参数梯度
    
    2. **vmap转换**: vmap(ft_compute_grad, in_dims=(None, 0, 0))
       - in_dims=(None, 0, 0) 表示:
         * params: None - 所有样本共享同一组参数
         * activations: 0 - 在第0维(batch维)上进行映射
         * backprops: 0 - 在第0维(batch维)上进行映射
    
    3. **批量处理**: 
       - 自动将batch中的每个样本分别传给基础函数
       - 并行计算每个样本的梯度
       - 将结果堆叠成 [batch_size, *param_shape] 的张量
    
    4. **等价操作**:
       for i in range(batch_size):
           grad_i = ft_compute_grad(params, activations[i], backprops[i])
       # vmap自动完成上述循环，但更高效
    """
    print(explanation)

def demonstrate_manual_vs_vmap():
    """演示手动循环 vs vmap的等价性"""
    print(f"\n" + "=" * 60)
    print("手动循环 vs vmap 等价性演示")
    print("=" * 60)
    
    # 创建简单的测试函数
    def single_sample_grad(params, activation, backprop):
        """模拟单个样本的梯度计算"""
        # 简化版：直接返回activation和backprop的外积作为"梯度"
        return torch.outer(backprop, activation)
    
    # 测试数据
    batch_size = 3
    activations = torch.randn(batch_size, 2)
    backprops = torch.randn(batch_size, 2)
    params = None  # 这个例子中不使用
    
    print(f"测试数据形状:")
    print(f"activations: {activations.shape}")
    print(f"backprops: {backprops.shape}")
    
    # 方法1: 手动循环
    manual_results = []
    for i in range(batch_size):
        grad_i = single_sample_grad(params, activations[i], backprops[i])
        manual_results.append(grad_i)
    manual_results = torch.stack(manual_results)
    
    # 方法2: 使用vmap
    vmap_func = vmap(single_sample_grad, in_dims=(None, 0, 0))
    vmap_results = vmap_func(params, activations, backprops)
    
    print(f"\n结果比较:")
    print(f"手动循环结果形状: {manual_results.shape}")
    print(f"vmap结果形状: {vmap_results.shape}")
    print(f"结果是否相等: {torch.allclose(manual_results, vmap_results)}")

if __name__ == "__main__":
    demonstrate_functorch_mechanism()
    test_ft_compute_sample_grad()
    explain_vmap_mechanism()
    demonstrate_manual_vs_vmap()
    
    print(f"\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)