# GPU Memory Profiler for Opacus

这个工具集提供了对Opacus训练过程中GPU内存使用的详细分析，包括不同模块（attention、activation、gradient等）的内存占用分析和可视化。

## 文件说明

1. **`gpu_memory_profiler.py`** - 基础内存分析工具
2. **`detailed_memory_profiler.py`** - 详细的模块级内存分析工具
3. **`requirements.txt`** - 依赖包列表

## 功能特性

### 基础内存分析 (`gpu_memory_profiler.py`)
- 测试不同参数组合下的内存使用情况
- 支持sequence length、batch size、LoRA rank等参数扫描
- 提供峰值内存使用统计
- 生成内存使用热力图和趋势图

### 详细内存分析 (`detailed_memory_profiler.py`)
- 模块级内存跟踪（attention、MLP、embedding等）
- 操作级内存分析（forward、backward、optimizer等）
- 梯度和激活内存估算
- 详细的内存分解可视化

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 基础内存分析

```bash
python gpu_memory_profiler.py \
    --token YOUR_HF_TOKEN \
    --seq_lengths 128 256 512 \
    --batch_sizes 8 16 32 \
    --max_physical_batch_sizes 1 2 4 \
    --lora_ranks 8 16 \
    --results_dir ./memory_results \
    --num_steps 5
```

### 2. 详细内存分析

```bash
python detailed_memory_profiler.py \
    --token YOUR_HF_TOKEN \
    --seq_lengths 128 256 \
    --batch_sizes 8 16 \
    --max_physical_batch_sizes 1 2 \
    --results_dir ./detailed_results \
    --num_steps 3
```

## 参数说明

### 通用参数
- `--token`: HuggingFace访问令牌（必需）
- `--results_dir`: 结果保存目录
- `--model_name`: 模型名称（默认: meta-llama/Llama-3.1-8B-Instruct）
- `--num_steps`: 每个配置的训练步数

### 测试参数
- `--seq_lengths`: 序列长度列表
- `--batch_sizes`: 批次大小列表
- `--max_physical_batch_sizes`: 最大物理批次大小列表
- `--lora_ranks`: LoRA秩列表（仅基础分析）

## 输出文件

### 基础分析输出
- `memory_profile_results.json`: 详细的JSON格式结果
- `memory_profile_summary.csv`: CSV格式的汇总数据
- `memory_analysis_overview.png`: 内存使用概览图
- `memory_timeline_config_*.png`: 每个配置的内存时间线
- `memory_heatmap.png`: 内存使用热力图

### 详细分析输出
- `detailed_memory_results.json`: 详细的模块级分析结果
- `detailed_memory_analysis.png`: 组件级内存分析图
- `operation_level_breakdown.png`: 操作级内存分解图

## 内存分析维度

### 1. 阶段分析
- 模型加载
- 数据集准备
- FSDP包装
- 优化器设置
- 隐私引擎设置
- 训练循环

### 2. 组件分析
- **Attention**: 注意力机制模块
- **MLP**: 多层感知机模块
- **Embedding**: 嵌入层
- **Normalization**: 归一化层
- **LoRA**: 低秩适应模块

### 3. 操作分析
- **Forward**: 前向传播
- **Backward**: 反向传播
- **Optimizer**: 优化器更新

### 4. 内存类型
- **Gradients**: 梯度内存
- **Activations**: 激活内存
- **Parameters**: 参数内存

## 使用建议

### 1. 参数选择
- 从小参数开始测试，避免内存溢出
- 优先测试关键参数组合
- 注意max_physical_batch_size <= batch_size

### 2. 结果分析
- 关注峰值内存使用情况
- 分析不同组件的内存占比
- 识别内存瓶颈模块

### 3. 优化策略
- 根据分析结果调整batch size
- 考虑使用gradient checkpointing
- 优化LoRA配置

## 注意事项

1. **GPU内存**: 确保有足够的GPU内存运行测试
2. **HuggingFace Token**: 需要有效的token访问gated模型
3. **多GPU**: 工具支持多GPU分布式训练
4. **结果解读**: 内存估算可能与实际使用有差异，仅供参考

## 故障排除

### 常见问题
1. **CUDA OOM**: 减少batch size或sequence length
2. **Token错误**: 检查HuggingFace token是否有效
3. **依赖缺失**: 安装所有必需的依赖包

### 性能优化
1. 使用较少的测试步数进行快速分析
2. 限制参数组合数量
3. 使用较小的模型进行初步测试

## 示例结果解读

### 内存分解示例
```
Component breakdown:
  attention: 2.34 GB
  mlp: 3.67 GB
  embedding: 0.89 GB
  lora: 0.45 GB
```

这表明MLP模块占用最多内存，可以考虑针对性优化。

### 参数影响分析
- Sequence length对内存影响呈二次增长
- Batch size对内存影响呈线性增长
- LoRA rank对内存影响相对较小

通过这些分析，可以更好地理解和优化Opacus训练的内存使用。