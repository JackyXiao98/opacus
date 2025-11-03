#!/usr/bin/env python3
"""
详细的内存分析报告：逐行分析原始方法和Triton方法的内存使用情况
针对用户指定的代码行进行详细分析
"""

import torch
import torch.nn as nn
import psutil
import gc
import sys
import os
from typing import Dict, Any, Tuple
from opt_einsum import contract

def get_memory_usage() -> float:
    """获取当前CPU内存使用量（MB）"""
    return psutil.Process().memory_info().rss / 1024 / 1024

def get_gpu_memory_usage() -> float:
    """获取当前GPU内存使用量（MB）"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0

def format_memory(mb: float) -> str:
    """格式化内存显示"""
    if mb >= 1024:
        return f"{mb/1024:.2f} GB"
    else:
        return f"{mb:.2f} MB"

def analyze_original_method_detailed(layer: nn.Linear, activations: torch.Tensor, gradients: torch.Tensor) -> Dict[str, Any]:
    """
    详细分析原始方法的内存使用情况
    对应 /Users/bytedance/Desktop/Github/opacus/memory_test/compare_linear_norm_methods.py#L67-77
    """
    print(f"\n{'='*100}")
    print("详细分析：原始方法 (compare_linear_norm_methods.py#L67-77)")
    print(f"{'='*100}")
    
    B, T, d = activations.shape
    p = layer.weight.shape[0]
    
    # 记录内存使用情况
    memory_log = []
    
    def log_memory(step: str):
        cpu_mem = get_memory_usage()
        gpu_mem = get_gpu_memory_usage()
        memory_log.append((step, cpu_mem, gpu_mem))
        print(f"    [{step}] CPU: {format_memory(cpu_mem)}, GPU: {format_memory(gpu_mem)}")
        return cpu_mem, gpu_mem
    
    # 开始分析
    log_memory("函数开始")
    
    # 第67行：activations = activations.to(dtype=torch.float32)
    print("\n第67行：activations = activations.to(dtype=torch.float32)")
    activations = activations.to(dtype=torch.float32)
    cpu_mem, gpu_mem = log_memory("dtype转换后")
    cpu_delta = cpu_mem - memory_log[0][1]
    gpu_delta = gpu_mem - memory_log[0][2]
    print(f"  内存变化：CPU {format_memory(cpu_delta)}, GPU {format_memory(gpu_delta)}")
    
    # 第68行：gradients = gradients.to(dtype=torch.float32)
    print("\n第68行：gradients = gradients.to(dtype=torch.float32)")
    gradients = gradients.to(dtype=torch.float32)
    cpu_mem, gpu_mem = log_memory("gradients dtype转换后")
    cpu_delta = cpu_mem - memory_log[1][1]
    gpu_delta = gpu_mem - memory_log[1][2]
    print(f"  内存变化：CPU {format_memory(cpu_delta)}, GPU {format_memory(gpu_delta)}")
    
    # 第70行：ggT = torch.bmm(gradients, gradients.transpose(-1, -2))
    print(f"\n第70行：ggT = torch.bmm(gradients, gradients.transpose(-1, -2))")
    print(f"  创建Gram矩阵 ggT: [{B}, {T}, {T}]")
    theoretical_ggT_size = B * T * T * 4 / (1024 * 1024)  # float32 = 4 bytes
    print(f"  理论内存需求：{format_memory(theoretical_ggT_size)}")
    
    ggT = torch.bmm(gradients, gradients.transpose(-1, -2))
    cpu_mem, gpu_mem = log_memory("ggT创建后")
    cpu_increase = cpu_mem - memory_log[-2][1]
    gpu_increase = gpu_mem - memory_log[-2][2]
    print(f"  实际内存增加：CPU {format_memory(cpu_increase)}, GPU {format_memory(gpu_increase)}")
    
    # 第71行：aaT = torch.bmm(activations, activations.transpose(-1, -2))
    print(f"\n第71行：aaT = torch.bmm(activations, activations.transpose(-1, -2))")
    print(f"  创建Gram矩阵 aaT: [{B}, {T}, {T}]")
    print(f"  理论内存需求：{format_memory(theoretical_ggT_size)}")
    
    aaT = torch.bmm(activations, activations.transpose(-1, -2))
    cpu_mem, gpu_mem = log_memory("aaT创建后")
    cpu_increase = cpu_mem - memory_log[-2][1]
    gpu_increase = gpu_mem - memory_log[-2][2]
    print(f"  实际内存增加：CPU {format_memory(cpu_increase)}, GPU {format_memory(gpu_increase)}")
    
    # 第73行：weight_norm_sample = torch.sqrt(torch.sum(ggT * aaT, dim=(1, 2)))
    print(f"\n第73行：weight_norm_sample = torch.sqrt(torch.sum(ggT * aaT, dim=(1, 2)))")
    print(f"  元素级乘法和求和操作")
    
    weight_norm_sample = torch.sqrt(torch.sum(ggT * aaT, dim=(1, 2)))
    cpu_mem, gpu_mem = log_memory("weight norm计算后")
    cpu_change = cpu_mem - memory_log[-2][1]
    gpu_change = gpu_mem - memory_log[-2][2]
    print(f"  内存变化：CPU {format_memory(cpu_change)}, GPU {format_memory(gpu_change)}")
    
    # 清理中间变量
    print(f"\n清理ggT和aaT矩阵")
    del ggT, aaT
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cpu_mem, gpu_mem = log_memory("清理ggT和aaT后")
    cpu_freed = memory_log[-2][1] - cpu_mem
    gpu_freed = memory_log[-2][2] - gpu_mem
    print(f"  释放内存：CPU {format_memory(cpu_freed)}, GPU {format_memory(gpu_freed)}")
    
    # 处理bias（如果存在）
    bias_norm_sample = None
    if layer.bias is not None:
        print(f"\n第75-76行：bias处理")
        print(f"第75行：ggT_bias = torch.bmm(gradients, gradients.transpose(-1, -2))")
        
        ggT_bias = torch.bmm(gradients, gradients.transpose(-1, -2))
        cpu_mem, gpu_mem = log_memory("bias ggT创建后")
        cpu_increase = cpu_mem - memory_log[-2][1]
        gpu_increase = gpu_mem - memory_log[-2][2]
        print(f"  内存增加：CPU {format_memory(cpu_increase)}, GPU {format_memory(gpu_increase)}")
        
        print(f"第76行：bias_norm_sample = torch.sqrt(torch.sum(ggT_bias, dim=(1, 2)))")
        bias_norm_sample = torch.sqrt(torch.sum(ggT_bias, dim=(1, 2)))
        cpu_mem, gpu_mem = log_memory("bias norm计算后")
        cpu_change = cpu_mem - memory_log[-2][1]
        gpu_change = gpu_mem - memory_log[-2][2]
        print(f"  内存变化：CPU {format_memory(cpu_change)}, GPU {format_memory(gpu_change)}")
        
        del ggT_bias
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cpu_mem, gpu_mem = log_memory("清理bias ggT后")
        cpu_freed = memory_log[-2][1] - cpu_mem
        gpu_freed = memory_log[-2][2] - gpu_mem
        print(f"  释放内存：CPU {format_memory(cpu_freed)}, GPU {format_memory(gpu_freed)}")
    
    # 打印内存使用总结
    print(f"\n{'='*80}")
    print("内存使用总结")
    print(f"{'='*80}")
    
    cpu_start = memory_log[0][1]
    cpu_end = memory_log[-1][1]
    gpu_start = memory_log[0][2]
    gpu_end = memory_log[-1][2]
    
    cpu_peak = max([log[1] for log in memory_log])
    gpu_peak = max([log[2] for log in memory_log])
    
    print(f"CPU内存:")
    print(f"  开始内存：{format_memory(cpu_start)}")
    print(f"  结束内存：{format_memory(cpu_end)}")
    print(f"  峰值内存：{format_memory(cpu_peak)}")
    print(f"  净内存变化：{format_memory(cpu_end - cpu_start)}")
    print(f"  峰值内存增加：{format_memory(cpu_peak - cpu_start)}")
    
    print(f"\nGPU内存:")
    print(f"  开始内存：{format_memory(gpu_start)}")
    print(f"  结束内存：{format_memory(gpu_end)}")
    print(f"  峰值内存：{format_memory(gpu_peak)}")
    print(f"  净内存变化：{format_memory(gpu_end - gpu_start)}")
    print(f"  峰值内存增加：{format_memory(gpu_peak - gpu_start)}")
    
    return {
        'weight': weight_norm_sample,
        'bias': bias_norm_sample,
        'memory_log': memory_log,
        'cpu_peak_memory': cpu_peak,
        'gpu_peak_memory': gpu_peak,
        'cpu_total_memory_change': cpu_end - cpu_start,
        'gpu_total_memory_change': gpu_end - gpu_start
    }

def analyze_triton_method_theoretical(B: int, T: int, d: int, p: int, tile_size: int = 256):
    """
    理论分析Triton方法的内存使用情况
    对应 /Users/bytedance/Desktop/Github/opacus/memory_test/triton_version/triton_flash_clipping_linear.py#L260-266
    """
    print(f"\n{'='*100}")
    print("理论分析：Triton方法 (triton_flash_clipping_linear.py#L260-266)")
    print(f"{'='*100}")
    
    # 分析每一行的内存需求
    print(f"\n第260行：if backprops.dim() == 2:")
    print(f"  条件判断，无内存开销")
    
    print(f"\n第261行：return contract('bi,bj->b', backprops, activations) ** 2")
    print(f"  2D情况：使用opt_einsum，内存需求最小")
    contract_memory = B * max(d, p) * 4 / (1024 * 1024)  # 临时数组
    print(f"  临时内存需求：{format_memory(contract_memory)}")
    
    print(f"\n第263-266行：3D情况的Triton优化")
    print(f"第263行：weight_norm_sq = _triton_frobenius_inner_over_T(activations, backprops, tile_size)")
    
    # 分析_triton_frobenius_inner_over_T的内存使用
    print(f"\n  _triton_frobenius_inner_over_T 内存分析：")
    print(f"  输入：activations [{B}, {T}, {d}], backprops [{B}, {T}, {p}]")
    print(f"  tile_size: {tile_size}")
    
    # 计算tile内存需求
    tile_A_memory = B * tile_size * d * 4 / (1024 * 1024)
    tile_G_memory = B * tile_size * p * 4 / (1024 * 1024)
    tile_result_memory = B * tile_size * tile_size * 4 / (1024 * 1024)
    
    print(f"  每个tile的内存需求：")
    print(f"    A_tile [{B}, {tile_size}, {d}]: {format_memory(tile_A_memory)}")
    print(f"    G_tile [{B}, {tile_size}, {p}]: {format_memory(tile_G_memory)}")
    print(f"    结果tile [{B}, {tile_size}, {tile_size}]: {format_memory(tile_result_memory)}")
    
    total_tile_memory = tile_A_memory + tile_G_memory + tile_result_memory
    print(f"  单个tile总内存：{format_memory(total_tile_memory)}")
    
    # 与原始方法对比
    original_gram_memory = B * T * T * 4 / (1024 * 1024)
    memory_savings = original_gram_memory / total_tile_memory
    
    print(f"\n第264行：if layer.bias is not None:")
    print(f"第265行：bias_norm_sq = _triton_sum_over_time_norm_squared(backprops, tile_size)")
    
    # bias计算的内存需求
    bias_tile_memory = B * tile_size * p * 4 / (1024 * 1024)
    print(f"  bias计算tile内存：{format_memory(bias_tile_memory)}")
    
    print(f"\n第266行：return {{'weight': torch.sqrt(weight_norm_sq), 'bias': torch.sqrt(bias_norm_sq) if layer.bias is not None else None}}")
    sqrt_memory = B * 4 / (1024 * 1024)  # 结果向量
    print(f"  sqrt操作内存：{format_memory(sqrt_memory)}")
    
    print(f"\n{'='*80}")
    print("Triton方法内存总结")
    print(f"{'='*80}")
    print(f"原始方法Gram矩阵内存：{format_memory(original_gram_memory)}")
    print(f"Triton方法tile内存：{format_memory(total_tile_memory)}")
    print(f"内存节省比例：{memory_savings:.1f}x")
    print(f"内存节省量：{format_memory(original_gram_memory - total_tile_memory)}")

def main():
    """主函数：执行详细的内存分析"""
    print("="*120)
    print("详细内存分析：逐行代码内存使用情况")
    print("="*120)
    
    # 设置参数（使用更小的矩阵进行测试）
    B, T, d, p = 4, 1024, 512, 512  # 使用更小的参数进行测试
    tile_size = 128
    
    print(f"测试参数：B={B}, T={T}, d={d}, p={p}")
    print(f"Tile size: {tile_size}")
    
    # 计算理论内存需求
    activations_memory = B * T * d * 4 / (1024 * 1024)
    gradients_memory = B * T * p * 4 / (1024 * 1024)
    original_gram_memory = B * T * T * 4 / (1024 * 1024)
    triton_tile_memory = B * tile_size * (d + p + tile_size) * 4 / (1024 * 1024)
    
    print(f"\n理论内存需求：")
    print(f"  Activations [{B}, {T}, {d}]: {format_memory(activations_memory)}")
    print(f"  Gradients [{B}, {T}, {p}]: {format_memory(gradients_memory)}")
    print(f"  原始方法Gram矩阵: {format_memory(original_gram_memory)}")
    print(f"  Triton方法tile内存: {format_memory(triton_tile_memory)}")
    print(f"  内存节省比例: {original_gram_memory/triton_tile_memory:.1f}x")
    
    try:
        # 创建测试数据
        print(f"\n创建测试数据...")
        # 优先使用GPU，如果不可用则使用CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        activations = torch.randn(B, T, d, device=device, dtype=torch.float32)
        gradients = torch.randn(B, T, p, device=device, dtype=torch.float32)
        
        # 创建线性层
        layer = nn.Linear(d, p, bias=True, device=device, dtype=torch.float32)
        
        print(f"数据创建完成，开始分析...")
        
        # 分析原始方法
        original_result = analyze_original_method_detailed(layer, activations, gradients)
        
        # 理论分析Triton方法
        analyze_triton_method_theoretical(B, T, d, p, tile_size)
        
        print(f"\n{'='*120}")
        print("分析完成")
        print(f"{'='*120}")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()