#!/usr/bin/env python3
"""
GRAD_SAMPLERS 和 NORM_SAMPLERS 注册机制详解

这个示例详细解释了Opacus中两个核心注册表的工作原理：
1. GRAD_SAMPLERS: 用于注册完整梯度计算函数
2. NORM_SAMPLERS: 用于注册梯度范数计算函数（Ghost Clipping）

重点解释第217行和第227行代码的作用机制。
"""

import torch
import torch.nn as nn
from opacus.grad_sample import GradSampleModule
from opacus.grad_sample.grad_sample_module_fast_gradient_clipping import GradSampleModuleFastGradientClipping

def explain_samplers_registry():
    """解释采样器注册表的基本概念"""
    print("=" * 80)
    print("GRAD_SAMPLERS 和 NORM_SAMPLERS 注册表解释")
    print("=" * 80)
    
    explanation = """
    Opacus使用两个注册表来管理不同层类型的梯度计算方法：
    
    📋 GRAD_SAMPLERS (梯度采样器注册表):
    --------------------------------
    • 作用: 存储每种层类型对应的完整per-sample梯度计算函数
    • 位置: GradSampleModule.GRAD_SAMPLERS
    • 用途: 计算完整的per-sample梯度 [batch_size, *param_shape]
    • 示例: nn.Linear → compute_linear_grad_sample()
    
    🚀 NORM_SAMPLERS (范数采样器注册表):
    ----------------------------------
    • 作用: 存储每种层类型对应的梯度范数计算函数
    • 位置: GradSampleModuleFastGradientClipping.NORM_SAMPLERS
    • 用途: 直接计算梯度范数 [batch_size] (Ghost Clipping)
    • 示例: nn.Linear → compute_linear_norm_sample()
    
    🔧 注册机制:
    -----------
    • 使用装饰器 @register_grad_sampler 和 @register_norm_sampler
    • 在模块导入时自动注册到对应的字典中
    • 支持一个函数注册到多个层类型
    """
    print(explanation)

def demonstrate_registry_content():
    """演示注册表的内容"""
    print(f"\n" + "=" * 80)
    print("注册表内容演示")
    print("=" * 80)
    
    # 查看GRAD_SAMPLERS的内容
    print("GRAD_SAMPLERS 注册的层类型:")
    for layer_type, sampler_func in GradSampleModule.GRAD_SAMPLERS.items():
        print(f"  {layer_type.__name__}: {sampler_func.__name__}")
    
    print(f"\nNORM_SAMPLERS 注册的层类型:")
    for layer_type, sampler_func in GradSampleModuleFastGradientClipping.NORM_SAMPLERS.items():
        print(f"  {layer_type.__name__}: {sampler_func.__name__}")

def explain_line_217_and_227():
    """详细解释第217行和第227行代码的作用"""
    print(f"\n" + "=" * 80)
    print("第217行和第227行代码详解")
    print("=" * 80)
    
    code_explanation = """
    这两行代码位于 capture_backprops_hook 方法中，负责选择合适的采样器函数：
    
    🎯 第217行: norm_sampler_fn = self.NORM_SAMPLERS[type(module)]
    --------------------------------------------------------
    • 位置: Ghost Clipping分支中
    • 作用: 从NORM_SAMPLERS注册表中获取当前模块类型对应的范数计算函数
    • 条件: self.use_ghost_clipping=True 且 type(module) in self.NORM_SAMPLERS
    • 结果: 直接计算梯度范数，无需完整梯度实例化
    
    🎯 第227行: grad_sampler_fn = self.GRAD_SAMPLERS[type(module)]
    ---------------------------------------------------------
    • 位置: Fast Gradient Clipping分支中
    • 作用: 从GRAD_SAMPLERS注册表中获取当前模块类型对应的梯度计算函数
    • 条件: 不使用Ghost Clipping 且 type(module) in self.GRAD_SAMPLERS
    • 结果: 计算完整per-sample梯度，然后转换为范数
    
    🔄 决策流程:
    -----------
    if self.use_ghost_clipping and type(module) in self.NORM_SAMPLERS:
        # 第217行: 使用Ghost Clipping (最高效)
        norm_sampler_fn = self.NORM_SAMPLERS[type(module)]
        norm_samples = norm_sampler_fn(module, activations, backprops)
    else:
        if not self.force_functorch and type(module) in self.GRAD_SAMPLERS:
            # 第227行: 使用专门的梯度采样器
            grad_sampler_fn = self.GRAD_SAMPLERS[type(module)]
        else:
            # 使用functorch通用方法
            grad_sampler_fn = ft_compute_per_sample_gradient
    """
    print(code_explanation)

def demonstrate_sampler_selection():
    """演示采样器选择过程"""
    print(f"\n" + "=" * 80)
    print("采样器选择过程演示")
    print("=" * 80)
    
    # 创建不同类型的模块
    modules = {
        "Linear": nn.Linear(10, 5),
        "Conv2d": nn.Conv2d(3, 16, 3),
        "ReLU": nn.ReLU(),
        "BatchNorm2d": nn.BatchNorm2d(16)
    }
    
    print("不同模块类型的采样器支持情况:")
    print(f"{'模块类型':<15} {'GRAD_SAMPLERS':<15} {'NORM_SAMPLERS':<15} {'选择策略'}")
    print("-" * 70)
    
    for name, module in modules.items():
        module_type = type(module)
        has_grad_sampler = module_type in GradSampleModule.GRAD_SAMPLERS
        has_norm_sampler = module_type in GradSampleModuleFastGradientClipping.NORM_SAMPLERS
        
        if has_norm_sampler:
            strategy = "Ghost Clipping (最优)"
        elif has_grad_sampler:
            strategy = "Fast Gradient Clipping"
        else:
            strategy = "Functorch (通用)"
        
        print(f"{name:<15} {'✅' if has_grad_sampler else '❌':<15} {'✅' if has_norm_sampler else '❌':<15} {strategy}")

def show_actual_sampler_functions():
    """展示实际的采样器函数"""
    print(f"\n" + "=" * 80)
    print("实际采样器函数示例")
    print("=" * 80)
    
    # 获取Linear层的采样器函数
    linear_grad_sampler = GradSampleModule.GRAD_SAMPLERS.get(nn.Linear)
    linear_norm_sampler = GradSampleModuleFastGradientClipping.NORM_SAMPLERS.get(nn.Linear)
    
    print("Linear层的采样器函数:")
    if linear_grad_sampler:
        print(f"  GRAD_SAMPLER: {linear_grad_sampler.__name__}")
        print(f"    文件位置: {linear_grad_sampler.__module__}")
        print(f"    函数签名: {linear_grad_sampler.__name__}(layer, activations, backprops)")
        print(f"    返回类型: Dict[nn.Parameter, torch.Tensor] (完整梯度)")
    
    if linear_norm_sampler:
        print(f"\n  NORM_SAMPLER: {linear_norm_sampler.__name__}")
        print(f"    文件位置: {linear_norm_sampler.__module__}")
        print(f"    函数签名: {linear_norm_sampler.__name__}(layer, activations, backprops)")
        print(f"    返回类型: Dict[nn.Parameter, torch.Tensor] (梯度范数)")

def explain_registration_process():
    """解释注册过程"""
    print(f"\n" + "=" * 80)
    print("采样器注册过程")
    print("=" * 80)
    
    registration_code = '''
    # 在 opacus/grad_sample/linear.py 中:
    
    from .utils import register_grad_sampler, register_norm_sampler
    
    @register_grad_sampler(nn.Linear)
    def compute_linear_grad_sample(layer, activations, backprops):
        """计算Linear层的完整per-sample梯度"""
        activations = activations[0]
        ret = {}
        if layer.weight.requires_grad:
            # 使用Einstein求和计算梯度: backprops ⊗ activations
            gs = torch.einsum("n...i,n...j->nij", backprops, activations)
            ret[layer.weight] = gs
        if layer.bias is not None and layer.bias.requires_grad:
            ret[layer.bias] = torch.einsum("n...k->nk", backprops)
        return ret
    
    @register_norm_sampler(nn.Linear)
    def compute_linear_norm_sample(layer, activations, backprops):
        """计算Linear层的梯度范数 (Ghost Clipping)"""
        activations = activations[0]
        ret = {}
        if layer.weight.requires_grad:
            # 直接计算范数: ||grad|| = sqrt(||backprops||² * ||activations||²)
            g = torch.einsum("n...i,n...i->n", backprops, backprops)
            a = torch.einsum("n...j,n...j->n", activations, activations)
            ret[layer.weight] = torch.sqrt((g * a).flatten())
        if layer.bias is not None and layer.bias.requires_grad:
            ret[layer.bias] = torch.sqrt(
                torch.einsum("n...i,n...i->n", backprops, backprops).flatten()
            )
        return ret
    '''
    
    print("注册装饰器的工作原理:")
    print(registration_code)
    
    print(f"\n注册装饰器实现 (在 utils.py 中):")
    decorator_code = '''
    def register_grad_sampler(target_class_or_classes):
        def decorator(f):
            for target_class in target_classes:
                GradSampleModule.GRAD_SAMPLERS[target_class] = f
                GradSampleModuleFastGradientClipping.GRAD_SAMPLERS[target_class] = f
            return f
        return decorator
    
    def register_norm_sampler(target_class_or_classes):
        def decorator(f):
            for target_class in target_classes:
                GradSampleModuleFastGradientClipping.NORM_SAMPLERS[target_class] = f
            return f
        return decorator
    '''
    print(decorator_code)

def demonstrate_performance_difference():
    """演示性能差异"""
    print(f"\n" + "=" * 80)
    print("性能差异对比")
    print("=" * 80)
    
    performance_comparison = """
    三种采样策略的性能对比:
    
    🚀 Ghost Clipping (第217行路径):
    ------------------------------
    • 内存使用: 最低 (只存储范数)
    • 计算速度: 最快 (专门优化的范数计算)
    • 支持层: 有限 (需要专门实现)
    • 适用场景: 生产环境，大模型训练
    
    ⚡ Fast Gradient Clipping (第227行路径):
    -------------------------------------
    • 内存使用: 中等 (临时存储梯度)
    • 计算速度: 中等 (需要完整梯度计算)
    • 支持层: 较多 (大部分常用层)
    • 适用场景: 平衡性能和兼容性
    
    🐌 Functorch (fallback路径):
    ---------------------------
    • 内存使用: 中等 (临时存储梯度)
    • 计算速度: 较慢 (通用实现)
    • 支持层: 所有层 (通用方法)
    • 适用场景: 兼容性优先，新层类型
    
    选择优先级: Ghost Clipping > Fast Gradient Clipping > Functorch
    """
    print(performance_comparison)

if __name__ == "__main__":
    print("GRAD_SAMPLERS 和 NORM_SAMPLERS 注册机制详解")
    print("=" * 80)
    
    explain_samplers_registry()
    demonstrate_registry_content()
    explain_line_217_and_227()
    demonstrate_sampler_selection()
    show_actual_sampler_functions()
    explain_registration_process()
    demonstrate_performance_difference()
    
    print(f"\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("关键要点:")
    print("• 第217行: 选择Ghost Clipping的范数计算函数 (最高效)")
    print("• 第227行: 选择传统的梯度计算函数 (兼容性好)")
    print("• 注册表通过装饰器在模块导入时自动填充")
    print("• 优先级: NORM_SAMPLERS > GRAD_SAMPLERS > functorch")
    print("• 这种设计实现了性能和兼容性的完美平衡")
    print("=" * 80)