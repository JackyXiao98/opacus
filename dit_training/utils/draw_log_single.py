import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

# 设置中文字体（避免中文乱码）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def parse_log_file(log_path: str) -> Tuple[List[int], List[float]]:
    """
    解析日志文件，提取 step 和 training loss
    Args:
        log_path: 日志文件路径
    Returns:
        (steps, losses): 步骤列表和对应的损失值列表
    """
    steps = []
    losses = []
    
    # 正则表达式：匹配 (step=xxxxxx) Train Loss: x.xxxx 格式
    pattern = r'\(step=(\d+)\) Train Loss: ([\d\.]+)'
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # 跳过 epoch 信息行
                if 'Beginning epoch' in line:
                    continue
                
                # 匹配目标数据
                match = re.search(pattern, line)
                if match:
                    step = int(match.group(1))  # 提取 step（自动去除前导零）
                    loss = float(match.group(2))  # 提取 loss
                    steps.append(step)
                    losses.append(loss)
                else:
                    # 忽略无法匹配的行（不报错，仅打印警告）
                    print(f"警告：第 {line_num} 行未匹配到数据，跳过：{line.strip()}")
    
    except FileNotFoundError:
        raise FileNotFoundError(f"错误：日志文件不存在 - {log_path}")
    except Exception as e:
        raise RuntimeError(f"错误：解析日志文件时出错 - {str(e)}")
    
    # 验证数据有效性
    if not steps or not losses:
        raise ValueError("错误：未从日志文件中提取到任何 step/loss 数据")
    if len(steps) != len(losses):
        raise ValueError(f"错误：step 数量（{len(steps)}）与 loss 数量（{len(losses)}）不匹配")
    
    print(f"成功提取 {len(steps)} 条有效数据")
    return steps, losses

def smooth_losses(losses: List[float], window_size: int = 5) -> List[float]:
    """
    对损失值进行滑动平均平滑（可选）
    Args:
        losses: 原始损失值列表
        window_size: 滑动窗口大小（需为奇数，默认5）
    Returns:
        平滑后的损失值列表
    """
    if window_size < 2 or window_size > len(losses):
        print(f"警告：滑动窗口大小 {window_size} 无效，使用原始数据")
        return losses
    
    # 确保窗口大小为奇数
    window_size = window_size if window_size % 2 == 1 else window_size + 1
    print(f"使用滑动窗口大小 {window_size} 进行损失平滑")
    
    # 滑动平均计算
    smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='same')
    # 处理边界（保持首尾数据不变）
    smoothed[:window_size//2] = losses[:window_size//2]
    smoothed[-window_size//2:] = losses[-window_size//2:]
    
    return smoothed.tolist()

def plot_step_loss(
    steps: List[int],
    losses: List[float],
    output_path: str = "step_loss_plot.png",
    title: str = "Training Loss vs Step",
    smooth_window: int = 0,
    dpi: int = 300,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    绘制 step vs training loss 趋势图
    Args:
        steps: 步骤列表
        losses: 损失值列表
        output_path: 图片输出路径（支持 png/jpg/svg 等格式）
        title: 图表标题
        smooth_window: 平滑窗口大小（0 表示不平滑）
        dpi: 图片分辨率
        figsize: 图片尺寸（宽，高）
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 可选：损失平滑
    if smooth_window > 1:
        losses_smoothed = smooth_losses(losses, smooth_window)
        # 绘制原始数据（浅色散点）
        ax.scatter(steps, losses, color='#94a3b8', alpha=0.5, s=10, label='Raw Loss')
        # 绘制平滑曲线（深色实线）
        ax.plot(steps, losses_smoothed, color='#1e40af', linewidth=2, label=f'Smoothed Loss (window={smooth_window})')
    else:
        # 绘制原始数据（线+点）
        ax.plot(steps, losses, color='#1e40af', linewidth=1.5, alpha=0.8, label='Training Loss')
        ax.scatter(steps, losses, color='#1e40af', s=20, alpha=0.6)
    
    # 设置图表样式
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Training Step', fontsize=12, fontweight='medium')
    ax.set_ylabel('Training Loss', fontsize=12, fontweight='medium')
    ax.set_ylim(0.1, 0.3)
    
    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)  # 网格在图层下方
    
    # 设置坐标轴样式
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1')
    ax.spines['bottom'].set_color('#cbd5e1')
    
    # 设置刻度标签大小
    ax.tick_params(axis='both', which='major', labelsize=10, colors='#475569')
    
    # 添加图例
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # 调整布局（避免标签被截断）
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"图表已保存到：{output_path}")
    
    # 可选：显示图片（本地运行时）
    try:
        plt.show()
    except Exception:
        print("提示：非本地环境，跳过图片显示")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="绘制 Training Loss vs Step 趋势图")
    parser.add_argument('--log', required=True, help="日志文件路径（必填）")
    parser.add_argument('--output', default="step_loss_plot.png", help="输出图片路径（默认：step_loss_plot.png）")
    parser.add_argument('--title', default="Training Loss vs Step (DP Training)", help="图表标题（默认：Training Loss vs Step）")
    parser.add_argument('--smooth', type=int, default=0, help="损失平滑窗口大小（默认0，不平滑；建议3-11之间的奇数）")
    parser.add_argument('--dpi', type=int, default=300, help="图片分辨率（默认300）")
    parser.add_argument('--figsize', nargs=2, type=int, default=[12, 6], help="图片尺寸（宽 高，默认：12 6）")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    try:
        # 1. 解析日志文件
        steps, losses = parse_log_file(args.log)
        
        # 2. 绘制图表
        plot_step_loss(
            steps=steps,
            losses=losses,
            output_path=args.output,
            title=args.title,
            smooth_window=args.smooth,
            dpi=args.dpi,
            figsize=tuple(args.figsize)
        )
        
        print("绘图完成！")
    
    except Exception as e:
        print(f"错误：{str(e)}")
        exit(1)