#!/bin/bash

# LLaMA内存使用可视化分析运行脚本

echo "=== LLaMA内存使用可视化分析 ==="
echo ""

# 检查CUDA是否可用
if command -v nvidia-smi &> /dev/null; then
    echo "检测到NVIDIA GPU:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    echo ""
else
    echo "未检测到NVIDIA GPU，将使用CPU模式"
    echo ""
fi

# 设置默认参数
MODEL_SIZE=${1:-"3b"}
BATCH_SIZE=${2:-1}
OUTPUT_FILE=${3:-"memory_analysis.png"}

echo "运行参数:"
echo "  模型大小: $MODEL_SIZE"
echo "  批次大小: $BATCH_SIZE"
echo "  输出文件: $OUTPUT_FILE"
echo "  序列长度: 64, 128, 256, 512, 1024, 2048"
echo ""

echo "开始内存分析..."
echo "注意: 这可能需要较长时间，特别是对于大模型和长序列"
echo ""

# 运行可视化分析
python memory_visualization.py \
    --model_size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --output $OUTPUT_FILE \
    --seq_lengths 64 128 256 512 1024 2048

echo ""
echo "分析完成！"
echo "图表已保存到: $OUTPUT_FILE"