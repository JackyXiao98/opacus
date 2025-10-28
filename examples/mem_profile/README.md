# 单GPU LLaMA内存分析工具

这个工具基于 `tensor_parallelism_llama.py` 创建，专门用于在单GPU环境下分析LLaMA模型在差分隐私训练过程中的内存使用情况。

## 功能特性

- 🔍 **详细内存分析**: 跟踪训练过程中每个阶段的GPU内存使用
- 📊 **多种模型大小**: 支持tiny、small、medium三种预设模型配置
- 🛡️ **差分隐私支持**: 集成Opacus差分隐私训练
- 📈 **实时监控**: 实时显示内存分配、释放和峰值使用情况
- ⚙️ **灵活配置**: 支持自定义批次大小、序列长度等参数

## 安装依赖

```bash
pip install -r requirements.txt
```

或者手动安装：

```bash
pip install torch>=2.0.0 transformers>=4.30.0 opacus>=1.4.0 numpy matplotlib psutil
```

## 快速开始

### 使用运行脚本（推荐）

```bash
# 使用默认参数（small模型，批次大小1，序列长度128，3个训练步骤）
./run_memory_analysis.sh

# 自定义参数
./run_memory_analysis.sh medium 2 256 5
# 参数顺序：模型大小 批次大小 序列长度 训练步数
```

### 直接运行Python脚本

```bash
# 基本用法
python single_gpu_memory_profiler.py

# 自定义参数
python single_gpu_memory_profiler.py \
    --model_size medium \
    --batch_size 2 \
    --seq_length 256 \
    --num_steps 5 \
    --learning_rate 1e-4 \
    --noise_multiplier 1.0 \
    --max_grad_norm 1.0
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_size` | `small` | 模型大小：`tiny`、`small`、`medium` |
| `--batch_size` | `1` | 批次大小 |
| `--seq_length` | `128` | 输入序列长度 |
| `--num_steps` | `3` | 训练步数 |
| `--learning_rate` | `1e-4` | 学习率 |
| `--noise_multiplier` | `1.0` | 差分隐私噪声倍数 |
| `--max_grad_norm` | `1.0` | 最大梯度范数（用于梯度裁剪） |
| `--device` | `cuda` | 计算设备（`cuda`或`cpu`） |

## 模型配置

### Tiny模型
- 词汇表大小: 1,000
- 隐藏层大小: 64
- 层数: 2
- 注意力头数: 4
- 适用于快速测试和调试

### Small模型
- 词汇表大小: 32,000
- 隐藏层大小: 512
- 层数: 4
- 注意力头数: 8
- 适用于中等规模实验

### Medium模型
- 词汇表大小: 32,000
- 隐藏层大小: 1,024
- 层数: 8
- 注意力头数: 16
- 适用于较大规模实验

### 3B模型
- 词汇表大小: 32,000
- 隐藏层大小: 3,200
- 层数: 26
- 注意力头数: 32
- 适用于大规模实验（约3B参数）

## 输出说明

工具会输出以下信息：

### 1. 系统信息
```
使用设备: cuda
模型大小: small
批次大小: 1
序列长度: 128
训练步数: 3
```

### 2. 模型配置
```
模型配置:
  词汇表大小: 32000
  隐藏层大小: 512
  层数: 4
  注意力头数: 8
```

### 3. 各阶段内存使用
```
=== Model Creation Memory Usage ===
Before: 0.00 MB allocated, 0.00 MB reserved
After:  245.67 MB allocated, 246.00 MB reserved
Peak:   245.67 MB allocated
Diff:   +245.67 MB allocated, +246.00 MB reserved
```

### 4. 训练过程内存分析
每个训练步骤包含以下阶段：
- **Forward Pass**: 前向传播
- **Loss Computation**: 损失计算
- **Backward Pass**: 反向传播
- **Optimizer Step**: 优化器更新
- **Zero Gradients**: 梯度清零

### 5. 内存使用总结
```
============================================================
MEMORY USAGE SUMMARY
============================================================
1. Model Creation
   Peak: 245.67 MB
   Diff: +245.67 MB
2. DP Setup
   Peak: 491.34 MB
   Diff: +245.67 MB
...
```

## 内存优化建议

### 减少内存使用
1. **减小批次大小**: 从较小的batch_size开始测试
2. **缩短序列长度**: 减少seq_length可显著降低内存使用
3. **使用更小的模型**: 从tiny模型开始，逐步增大
4. **启用梯度检查点**: 在实际应用中考虑使用gradient checkpointing

### 内存不足时的处理
1. 减少批次大小到1
2. 减少序列长度到64或更小
3. 使用tiny模型配置
4. 考虑使用CPU模式（虽然会很慢）

## 与原始tensor_parallelism_llama.py的区别

| 特性 | 原始文件 | 单GPU版本 |
|------|----------|-----------|
| 分布式训练 | ✅ 支持tensor parallelism | ❌ 单GPU训练 |
| 内存分析 | ✅ 基础profile_mem函数 | ✅ 详细的GPUMemoryProfiler类 |
| 模型配置 | ❌ 固定配置 | ✅ 多种预设配置 |
| 命令行参数 | ❌ 无 | ✅ 完整的argparse支持 |
| 文档说明 | ❌ 最少 | ✅ 详细的README和注释 |
| 错误处理 | ❌ 基础 | ✅ 更好的错误处理和提示 |

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   解决方案：减少batch_size或seq_length

2. **CUDA不可用**
   ```
   CUDA不可用，切换到CPU
   ```
   工具会自动切换到CPU模式，但运行速度会很慢

3. **依赖缺失**
   ```
   ModuleNotFoundError: No module named 'opacus'
   ```
   解决方案：安装requirements.txt中的依赖

### 性能优化

1. **预热GPU**: 第一次运行可能较慢，后续运行会更快
2. **监控系统资源**: 使用`nvidia-smi`监控GPU使用情况
3. **调整参数**: 根据硬件配置调整模型大小和批次大小

## 内存使用可视化

### 功能特性
工具包含一个强大的可视化组件，可以分析不同序列长度下的内存使用情况：

- 📊 **多序列长度分析**: 支持64, 128, 256, 512, 1024, 2048等序列长度
- 📈 **分阶段可视化**: 为每个训练阶段生成独立的内存使用图表
- 🎯 **3B模型支持**: 专门优化用于分析大规模模型
- 📋 **数据表格**: 生成详细的内存使用总结表格

### 使用方法

#### 快速开始
```bash
# 使用默认参数（3B模型，所有序列长度）
./run_visualization.sh

# 自定义参数
./run_visualization.sh 3b 1 memory_3b_analysis.png
```

#### 详细配置
```bash
python memory_visualization.py \
    --model_size 3b \
    --batch_size 1 \
    --seq_lengths 64 128 256 512 1024 2048 \
    --output memory_analysis.png
```

### 可视化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_size` | `3b` | 模型大小：`tiny`、`small`、`medium`、`3b` |
| `--batch_size` | `1` | 批次大小 |
| `--seq_lengths` | `[64,128,256,512,1024,2048]` | 要分析的序列长度列表 |
| `--output` | `memory_analysis.png` | 输出图片文件路径 |
| `--device` | `cuda` | 计算设备 |

### 输出说明

#### 可视化图表
生成的图表包含6个子图，每个对应一个训练阶段：
1. **Model Creation**: 模型创建阶段
2. **DP Setup**: 差分隐私设置阶段  
3. **Forward Pass**: 前向传播阶段
4. **Backward Pass**: 反向传播阶段
5. **Optimizer Step**: 优化器更新阶段
6. **Zero Gradients**: 梯度清零阶段

每个子图显示：
- X轴：序列长度（对数刻度）
- Y轴：内存差异（MB）
- 数据点：每个序列长度的内存使用量
- 趋势线：显示内存使用随序列长度的变化趋势

#### 数据表格
控制台输出详细的内存使用表格，包含：
- 各个阶段的内存使用情况
- 不同序列长度的对比数据
- 便于进一步分析的数值数据

### 使用建议

1. **从小模型开始**: 先用`tiny`或`small`模型测试，确保环境正常
2. **逐步增加序列长度**: 避免一次性测试过长的序列导致内存溢出
3. **监控系统资源**: 使用`nvidia-smi`监控GPU内存使用情况
4. **保存结果**: 生成的图表可用于论文、报告或进一步分析

## 扩展功能

可以基于此工具进行以下扩展：

1. **添加更多模型**: 支持其他Transformer模型
2. **多算法对比**: 在同一图表中比较不同的DP算法
3. **基准测试**: 与其他框架进行性能对比
4. **自动调优**: 自动寻找最优的内存配置
5. **分布式版本**: 扩展到多GPU环境

## 许可证

本工具继承原始文件的Apache 2.0许可证。

## 贡献

欢迎提交Issue和Pull Request来改进这个工具！