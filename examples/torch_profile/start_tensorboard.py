#!/usr/bin/env python3
"""
启动 TensorBoard 来查看 PyTorch Profiler 数据

这个脚本确保 TensorBoard 正确加载 PyTorch Profiler 插件
"""

import subprocess
import sys
import os

def check_tensorboard_pytorch_profiler():
    """检查是否安装了 TensorBoard PyTorch Profiler 插件"""
    try:
        import torch_tb_profiler
        print("✓ torch-tb-profiler 已安装")
        return True
    except ImportError:
        print("✗ torch-tb-profiler 未安装")
        return False

def install_pytorch_profiler_plugin():
    """安装 TensorBoard PyTorch Profiler 插件"""
    print("正在安装 torch-tb-profiler...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-tb-profiler"])
        print("✓ torch-tb-profiler 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 安装失败: {e}")
        return False

def start_tensorboard():
    """启动 TensorBoard"""
    runs_dir = "./runs"
    
    if not os.path.exists(runs_dir):
        print(f"✗ 目录 {runs_dir} 不存在")
        return False
    
    # 检查是否有 trace 文件
    trace_files = []
    for root, dirs, files in os.walk(runs_dir):
        for file in files:
            if file.endswith('.pt.trace.json'):
                trace_files.append(os.path.join(root, file))
    
    if not trace_files:
        print(f"✗ 在 {runs_dir} 中没有找到 .pt.trace.json 文件")
        return False
    
    print(f"✓ 找到 {len(trace_files)} 个 trace 文件")
    
    # 启动 TensorBoard
    print(f"启动 TensorBoard，日志目录: {runs_dir}")
    print("请在浏览器中打开: http://localhost:6006")
    print("在 TensorBoard 中，点击 'PYTORCH_PROFILER' 标签页查看性能数据")
    print("\n按 Ctrl+C 停止 TensorBoard")
    
    try:
        subprocess.run([
            "tensorboard", 
            "--logdir", runs_dir,
            "--port", "6006",
            "--host", "localhost"
        ])
    except KeyboardInterrupt:
        print("\nTensorBoard 已停止")
    except FileNotFoundError:
        print("✗ tensorboard 命令未找到，请确保已安装 tensorboard")
        print("安装命令: pip install tensorboard")
        return False
    
    return True

def main():
    print("=== TensorBoard PyTorch Profiler 启动器 ===\n")
    
    # 检查并安装 PyTorch Profiler 插件
    if not check_tensorboard_pytorch_profiler():
        print("需要安装 torch-tb-profiler 插件来查看 PyTorch Profiler 数据")
        if input("是否现在安装? (y/n): ").lower().startswith('y'):
            if not install_pytorch_profiler_plugin():
                print("安装失败，请手动运行: pip install torch-tb-profiler")
                return
        else:
            print("请手动安装: pip install torch-tb-profiler")
            return
    
    # 启动 TensorBoard
    start_tensorboard()

if __name__ == "__main__":
    main()