import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict
import os

# 设置中文字体（避免中文乱码）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 预定义配色方案（支持最多10个日志文件，可扩展）
COLORS = ['#1e40af', '#dc2626', '#059669', '#d97706', '#7c3aed', 
          '#14b8a6', '#db2777', '#64748b', '#0891b2', '#f59e0b']
# 预定义线型（循环使用）
LINE_STYLES = ['-', '--', '-.', ':']
# 预定义标记（循环使用）
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

def parse_log_file(log_path: str) -> Tuple[List[int], List[float]]:
    """
    解析单个日志文件，提取 step 和 training loss
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
                # 忽略无法匹配的行（不报错，仅打印警告）
    
    except FileNotFoundError:
        raise FileNotFoundError(f"错误：日志文件不存在 - {log_path}")
    except Exception as e:
        raise RuntimeError(f"错误：解析日志文件 {log_path} 时出错 - {str(e)}")
    
    # 验证数据有效性
    if not steps or not losses:
        raise ValueError(f"错误：未从日志文件 {log_path} 中提取到任何 step/loss 数据")
    if len(steps) != len(losses):
        raise ValueError(f"错误：日志文件 {log_path} 的 step 数量（{len(steps)}）与 loss 数量（{len(losses)}）不匹配")
    
    print(f"成功从 {os.path.basename(log_path)} 提取 {len(steps)} 条有效数据")
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
        return losses
    
    # 确保窗口大小为奇数
    window_size = window_size if window_size % 2 == 1 else window_size + 1
    
    # 滑动平均计算
    smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='same')
    # 处理边界（保持首尾数据不变）
    smoothed[:window_size//2] = losses[:window_size//2]
    smoothed[-window_size//2:] = losses[-window_size//2:]
    
    return smoothed.tolist()

def plot_multi_logs(
    log_paths: List[str],
    log_labels: List[str] = None,
    output_path: str = "multi_step_loss_plot.png",
    title: str = "Training Loss vs Step (Multi-Log Comparison)",
    smooth_window: int = 0,
    dpi: int = 300,
    figsize: Tuple[int, int] = (12, 6),
    show_raw: bool = False
) -> None:
    """
    绘制多个日志文件的 step vs training loss 对比图
    Args:
        log_paths: 日志文件路径列表
        log_labels: 每个日志的自定义标签（默认使用文件名）
        output_path: 图片输出路径（支持 png/jpg/svg 等格式）
        title: 图表标题
        smooth_window: 损失平滑窗口大小（0 表示不平滑）
        dpi: 图片分辨率
        figsize: 图片尺寸（宽，高）
        show_raw: 是否同时显示原始数据（仅平滑时生效）
    """
    # 校验日志文件数量
    if len(log_paths) == 0:
        raise ValueError("错误：至少需要传入一个日志文件")
    if len(log_paths) > len(COLORS):
        print(f"警告：日志文件数量（{len(log_paths)}）超过预定义颜色数量（{len(COLORS)}），将重复使用颜色")
    
    # 处理标签（默认使用文件名，去除后缀）
    if log_labels is None:
        log_labels = [os.path.splitext(os.path.basename(path))[0] for path in log_paths]
    else:
        if len(log_labels) != len(log_paths):
            raise ValueError(f"错误：标签数量（{len(log_labels)}）与日志文件数量（{len(log_paths)}）不匹配")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 遍历每个日志文件，绘制曲线
    for idx, (log_path, label) in enumerate(zip(log_paths, log_labels)):
        # 解析日志数据
        steps, losses = parse_log_file(log_path)
        
        # 可选：损失平滑
        if smooth_window > 1:
            losses_processed = smooth_losses(losses, smooth_window)
            # 选择颜色、线型、标记
            color = COLORS[idx % len(COLORS)]
            line_style = LINE_STYLES[idx % len(LINE_STYLES)]
            marker = MARKERS[idx % len(MARKERS)]
            
            # 绘制平滑曲线（主曲线）
            ax.plot(
                steps, losses_processed,
                color=color, linewidth=2, linestyle=line_style,
                label=f'{label} (smoothed, window={smooth_window})',
                alpha=0.9
            )
            
            # 可选：显示原始数据（浅色散点）
            if show_raw:
                ax.scatter(
                    steps, losses,
                    color=color, s=15, alpha=0.4, marker=marker,
                    label=f'{label} (raw)' if idx == 0 else ""  # 仅第一个日志显示raw标签，避免图例冗余
                )
        else:
            # 不平滑：绘制原始数据（线+点）
            color = COLORS[idx % len(COLORS)]
            line_style = LINE_STYLES[idx % len(LINE_STYLES)]
            marker = MARKERS[idx % len(MARKERS)]
            
            ax.plot(
                steps, losses,
                color=color, linewidth=1.5, linestyle=line_style,
                alpha=0.8, label=label
            )
            ax.scatter(
                steps, losses,
                color=color, s=20, alpha=0.6, marker=marker
            )
    
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
    
    # 添加图例（自动调整位置，避免遮挡）
    ax.legend(loc='best', fontsize=10, framealpha=0.9, ncol=2 if len(log_paths) > 3 else 1)
    
    # 调整布局（避免标签被截断）
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"\n对比图表已保存到：{output_path}")
    
    # 可选：显示图片（本地运行时）
    try:
        plt.show()
    except Exception:
        print("提示：非本地环境，跳过图片显示")



''''
python /mnt/bn/watermark/split_volume/zhaoyuchen/Project/diffusion-model-dp-training/utils/draw_log_multi.py --logs /mnt/bn/watermark/split_volume/zhaoyuchen/Dataset/dit-results/dp-DiT-B-4-img512-cls1-bs256-noise0.3701-ffhq/001-DiT-B-4/log.txt /mnt/bn/watermark/split_volume/zhaoyuchen/Dataset/dit-results/non-dp-DiT-B-4-img512-cls1-bs256-ffhq/001-DiT-B-4/log.txt --labels dp non-dp
'''
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="绘制多个日志的 Training Loss vs Step 对比图")
    parser.add_argument('--logs', required=True, nargs='+', help="日志文件路径列表（必填，支持多个文件）")
    parser.add_argument('--labels', nargs='+', help="每个日志的自定义标签（可选，数量需与日志文件一致）")
    parser.add_argument('--output', default="multi_step_loss_plot.png", help="输出图片路径（默认：multi_step_loss_plot.png）")
    parser.add_argument('--title', default="Training Loss vs Step (Multi-Log Comparison)", help="图表标题")
    parser.add_argument('--smooth', type=int, default=0, help="损失平滑窗口大小（默认0，不平滑；建议3-11之间的奇数）")
    parser.add_argument('--show-raw', action='store_true', help="平滑模式下是否显示原始数据（默认不显示）")
    parser.add_argument('--dpi', type=int, default=300, help="图片分辨率（默认300）")
    parser.add_argument('--figsize', nargs=2, type=int, default=[12, 6], help="图片尺寸（宽 高，默认：12 6）")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    try:
        # 绘制多日志对比图
        plot_multi_logs(
            log_paths=args.logs,
            log_labels=args.labels,
            output_path=args.output,
            title=args.title,
            smooth_window=args.smooth,
            dpi=args.dpi,
            figsize=tuple(args.figsize),
            show_raw=args.show_raw
        )
        
        print("多日志对比绘图完成！")
    
    except Exception as e:
        print(f"错误：{str(e)}")
        exit(1)