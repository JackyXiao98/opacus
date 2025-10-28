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
import os
import sys
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict

# 添加当前目录到Python路径，以便导入single_gpu_memory_profiler
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from single_gpu_memory_profiler import run_memory_analysis


def collect_memory_data(sequence_lengths: List[int], model_size: str = "3b", 
                       batch_size: int = 1, device: str = "cuda") -> Dict[str, Dict[int, float]]:
    """收集不同序列长度下的内存使用数据"""
    print(f"开始收集内存数据...")
    print(f"模型大小: {model_size}")
    print(f"批次大小: {batch_size}")
    print(f"序列长度: {sequence_lengths}")
    print(f"设备: {device}")
    
    # 存储结果: {stage_name: {seq_length: memory_diff}}
    memory_data = defaultdict(dict)
    
    for seq_len in sequence_lengths:
        print(f"\n正在分析序列长度: {seq_len}")
        
        try:
            # 运行内存分析
            stats = run_memory_analysis(
                model_size=model_size,
                batch_size=batch_size,
                seq_length=seq_len,
                device=device,
                verbose=True
            )
            
            # 提取每个阶段的内存差异
            for stat in stats:
                stage_name = stat['stage']
                memory_diff = stat['diff']['allocated_diff']
                memory_data[stage_name][seq_len] = memory_diff
                
        except Exception as e:
            print(f"序列长度 {seq_len} 分析失败: {e}")
            # 如果失败，设置为0或跳过
            continue
    
    return dict(memory_data)


def create_visualization(memory_data: Dict[str, Dict[int, float]], 
                        sequence_lengths: List[int],
                        output_path: str = "memory_analysis.png",
                        model_size: str = "3b"):
    """创建内存使用可视化图表"""
    
    # 定义要显示的阶段和颜色
    stage_colors = {
        'Model Creation': '#1f77b4',
        'DP Setup': '#ff7f0e', 
        'Forward Pass': '#2ca02c',
        'Backward Pass': '#d62728',
        'Optimizer Step': '#9467bd',
        'Zero Gradients': '#8c564b'
    }
    
    # 过滤出有数据的阶段
    available_stages = [stage for stage in stage_colors.keys() if stage in memory_data]
    
    if not available_stages:
        print("没有可用的内存数据进行可视化")
        return
    
    # 创建子图
    n_stages = len(available_stages)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Memory Usage Analysis - {model_size.upper()} Model\nMemory Difference vs Sequence Length', 
                 fontsize=16, fontweight='bold')
    
    # 展平axes数组以便索引
    axes_flat = axes.flatten()
    
    for i, stage in enumerate(available_stages):
        ax = axes_flat[i]
        
        # 获取该阶段的数据
        stage_data = memory_data[stage]
        
        # 准备数据
        x_values = []
        y_values = []
        
        for seq_len in sequence_lengths:
            if seq_len in stage_data:
                x_values.append(seq_len)
                y_values.append(stage_data[seq_len])
        
        if x_values and y_values:
            # 绘制线图
            ax.plot(x_values, y_values, 'o-', 
                   color=stage_colors[stage], 
                   linewidth=2, 
                   markersize=8,
                   label='Opacus DP')
            
            # 设置图表属性
            ax.set_title(f'{stage}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Sequence Length', fontsize=10)
            ax.set_ylabel('Memory Diff (MB)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 设置x轴刻度
            ax.set_xscale('log', base=2)
            ax.set_xticks(x_values)
            ax.set_xticklabels([str(x) for x in x_values])
            
            # 添加数值标签
            for x, y in zip(x_values, y_values):
                ax.annotate(f'{y:.1f}', (x, y), 
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center', 
                           fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No Data Available', 
                   transform=ax.transAxes, 
                   ha='center', va='center',
                   fontsize=12, color='red')
            ax.set_title(f'{stage}', fontsize=12, fontweight='bold')
    
    # 隐藏多余的子图
    for i in range(len(available_stages), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n可视化图表已保存到: {output_path}")
    
    return fig


def create_summary_table(memory_data: Dict[str, Dict[int, float]], 
                        sequence_lengths: List[int]):
    """创建内存使用总结表格"""
    print("\n" + "="*80)
    print("MEMORY USAGE SUMMARY TABLE")
    print("="*80)
    
    # 表头
    header = "Stage".ljust(20)
    for seq_len in sequence_lengths:
        header += f"{seq_len}".rjust(10)
    print(header)
    print("-" * 80)
    
    # 数据行
    for stage, stage_data in memory_data.items():
        row = stage.ljust(20)
        for seq_len in sequence_lengths:
            if seq_len in stage_data:
                row += f"{stage_data[seq_len]:.1f}".rjust(10)
            else:
                row += "N/A".rjust(10)
        print(row)
    
    print("="*80)
    print("单位: MB (内存差异)")


def main():
    parser = argparse.ArgumentParser(description="LLaMA内存使用可视化工具")
    parser.add_argument("--model_size", choices=["tiny", "small", "medium", "1b", "3b"], 
                       default="3b", help="模型大小")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--device", default="cuda", help="计算设备")
    parser.add_argument("--output", default="memory_analysis.png", help="输出图片路径")
    parser.add_argument("--seq_lengths", nargs='+', type=int, 
                       default=[64, 128, 256, 512, 1024, 2048],
                       help="要测试的序列长度列表")
    
    args = parser.parse_args()
    
    print("="*60)
    print("LLaMA内存使用可视化分析")
    print("="*60)
    
    # 收集内存数据
    memory_data = collect_memory_data(
        sequence_lengths=args.seq_lengths,
        model_size=args.model_size,
        batch_size=args.batch_size,
        device=args.device
    )
    
    if not memory_data:
        print("没有收集到内存数据，退出程序")
        return
    
    # 创建可视化
    fig = create_visualization(
        memory_data=memory_data,
        sequence_lengths=args.seq_lengths,
        output_path=args.output,
        model_size=args.model_size
    )
    
    # 创建总结表格
    create_summary_table(memory_data, args.seq_lengths)
    
    print(f"\n分析完成！")
    print(f"图表保存在: {args.output}")


if __name__ == "__main__":
    main()