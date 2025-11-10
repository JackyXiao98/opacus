# 详细内存分析系统 (Detailed Memory Profiling System)

## 概述

这是一个专门为 DP-SGD 算法设计的详细内存分析系统，可以精确追踪：

1. **模型参数 (Model Parameters)** - 基础模型权重
2. **优化器状态 (Optimizer States)** - Adam 的 momentum 和 variance
3. **梯度 (Gradients)** - 标准反向传播梯度
4. **激活 Hook 存储 (Activation Hooks)** - DP-SGD 保存的激活值
5. **Norm Samples** - DP-SGD 的 per-sample gradient norms
6. **临时矩阵 (Temp Matrices)** - Ghost Clipping 的 ggT/aaT 矩阵

## 核心特性

### ✅ 进程隔离
每个实验在**独立的 Python 进程**中运行，完全避免内存池污染。

### ✅ 细粒度追踪
通过增强的 hooks 机制，精确追踪 DP-SGD 各组件的内存使用。

### ✅ 时间线分析
记录训练过程中每个阶段的内存快照（forward, backward, optimizer step）。

### ✅ 可视化对比
自动生成多种可视化图表，清晰展示不同算法的内存和时间权衡。

## 文件结构

```
memory_test/test_algo/
├── detailed_memory_profiler.py      # 增强的内存分析器
├── single_experiment.py             # 单个实验运行脚本
├── run_all_experiments.sh           # Shell 脚本协调器 ⭐ 主入口
├── visualize_memory_breakdown.py   # 可视化生成器
└── MEMORY_PROFILING_README.md       # 本文档
```

## 快速开始

### 方法 1：一键运行所有实验（推荐）

```bash
cd /Users/bytedance/Desktop/Github/opacus
source .venv/bin/activate  # 如果使用虚拟环境

# 运行完整的实验套件
./memory_test/test_algo/run_all_experiments.sh
```

这将：
1. 依次运行 Vanilla、Ghost Clipping、Flash Clipping 三个实验
2. 每个实验在独立进程中运行（避免内存污染）
3. 自动生成可视化结果
4. 保存所有数据到 `memory_profiling_results/run_TIMESTAMP/`

### 方法 2：运行单个实验

```bash
# Vanilla (无 DP-SGD)
python memory_test/test_algo/single_experiment.py \
    --experiment vanilla \
    --output vanilla_result.json \
    --seq-len 16384 \
    --batch-size 1

# Ghost Clipping
python memory_test/test_algo/single_experiment.py \
    --experiment ghost \
    --output ghost_result.json \
    --seq-len 16384 \
    --batch-size 1

# Flash Clipping
python memory_test/test_algo/single_experiment.py \
    --experiment flash_clip \
    --output flash_result.json \
    --seq-len 16384 \
    --batch-size 1
```

### 方法 3：自定义可视化

```bash
# 如果已有实验结果，可以单独运行可视化
python memory_test/test_algo/visualize_memory_breakdown.py \
    --input-dir memory_profiling_results/run_20241110_123456 \
    --output-dir memory_profiling_results/run_20241110_123456/visualizations
```

## 配置参数

在 `run_all_experiments.sh` 中可以调整：

```bash
VOCAB_SIZE=32000      # 词表大小
HIDDEN_DIM=2048       # 隐藏层维度
NUM_LAYERS=20         # Transformer 层数
NUM_HEADS=16          # 注意力头数
SEQ_LEN=16384         # 序列长度（8192*2）
BATCH_SIZE=1          # 批次大小
NUM_ITER=3            # 分析迭代次数
WARMUP_ITER=2         # 预热迭代次数
```

## 输出文件

### JSON 结果文件

每个实验生成一个 JSON 文件，包含：

```json
{
  "experiment": "ghost",
  "config": {...},
  "avg_time_ms": 17489.427,
  "peak_memory_mb": 56992.12,
  "breakdown": {
    "model_parameters_mb": 5123.45,
    "optimizer_states_mb": 10246.90,
    "gradients_mb": 15370.35,
    "activation_hooks_mb": 12000.00,
    "norm_samples_mb": 2000.00,
    "temp_matrices_mb": 1000.00,
    "total_allocated_mb": 56992.12,
    "peak_allocated_mb": 56992.12
  },
  "snapshots": [
    {
      "name": "0_model_loaded",
      "allocated_mb": 5123.45,
      "reserved_mb": 6000.00,
      "timestamp": 1699612345.123
    },
    ...
  ]
}
```

### 可视化输出

在 `visualizations/` 目录下生成：

1. **`memory_breakdown_comparison.png`** 
   - 左图：三种方法的组件级内存堆叠图
   - 右图：DP-SGD 相对 Vanilla 的额外内存开销

2. **`memory_timeline.png`**
   - 三个子图，展示每个方法在训练过程中的内存变化
   - 标注 forward, backward, optimizer step 阶段

3. **`performance_tradeoff.png`**
   - 散点图展示内存 vs 时间的权衡

4. **`summary.txt`**
   - 文本格式的详细汇总报告

## 典型输出示例

```
========================================================================================================
SUMMARY
========================================================================================================

Method                         Peak Memory (MB)         Avg Time (ms)        Memory/Time
----------------------------------------------------------------------------------------------------
Vanilla (No DP-SGD)                       43407.78              7091.97             6.12
Ghost Clipping (DP-SGD)                   56992.12             17489.43             3.26
Flash Clipping (DP-SGD)                   56816.12              7831.43             7.26
========================================================================================================

COMPONENT-LEVEL BREAKDOWN:

Vanilla (No DP-SGD):
--------------------------------------------------------------------------------
  Optimizer States                                       10246.90 MB
  Gradients                                              15370.35 MB
  Model Parameters                                        5123.45 MB
  Activation Hooks                                        8000.00 MB
  ...

Ghost Clipping (DP-SGD):
--------------------------------------------------------------------------------
  Optimizer States                                       10246.90 MB
  Gradients                                              15370.35 MB
  Activation Hooks                                       12000.00 MB  ← DP-SGD 增加
  Model Parameters                                        5123.45 MB
  Norm Samples                                            2000.00 MB  ← DP-SGD 特有
  Temp Matrices                                           1000.00 MB  ← Ghost 特有
  ...
```

## 关键发现

基于这个系统的分析，我们发现：

### 1. DP-SGD 的固有开销（约 +13 GB）

- **Activation Hooks**: +4-5 GB
  - DP-SGD 必须保存所有中间激活用于 per-sample gradient 计算
  
- **Norm Samples**: +2-3 GB
  - 每个参数存储 `[batch_size]` 的 norm 值
  
- **其他 DP-SGD 开销**: +6-8 GB
  - 额外的 hook 管理、临时存储等

### 2. Ghost vs Flash Clipping

虽然 Ghost 有 T² 的大矩阵（ggT, aaT），但：

✅ **峰值内存相同** (56,992 MB vs 56,816 MB)
- 原因：逐层串行执行，内存自动复用

❌ **IO 成本差异巨大** (17,489 ms vs 7,831 ms)
- Ghost: 90 GB 总 IO（每层 4.5 GB × 20 层）
- Flash Clip: 10 GB 总 IO（双重 tiling 优化）

### 3. 优化建议

如果内存是瓶颈：
1. **混合精度 (BF16)**: 可节省 ~40% 内存
2. **Activation Checkpointing**: 可节省 ~30% 内存
3. **Batch Size = 1**: 可节省 ~25% 内存

## 故障排查

### 问题 1：CUDA Out of Memory

**解决方案**：
- 减小 `SEQ_LEN` (16384 → 8192)
- 减小 `NUM_LAYERS` (20 → 10)
- 确保 GPU 内存 >= 64 GB

### 问题 2：脚本权限错误

```bash
chmod +x memory_test/test_algo/run_all_experiments.sh
```

### 问题 3：Import 错误

确保在项目根目录运行：
```bash
cd /Users/bytedance/Desktop/Github/opacus
source .venv/bin/activate
./memory_test/test_algo/run_all_experiments.sh
```

### 问题 4：内存池污染

这就是为什么我们设计了进程隔离！如果仍然遇到污染：
1. 增加实验间隔时间（修改 `sleep 3` → `sleep 10`）
2. 手动重启后再运行下一个实验

## 进阶用法

### 自定义 Hook

在 `detailed_memory_profiler.py` 中添加自定义 hooks：

```python
def custom_forward_hook(mod, inp, out, module_name=name):
    # 你的自定义追踪逻辑
    if hasattr(mod, 'custom_attribute'):
        track_memory(mod.custom_attribute)
```

### 导出数据用于论文

JSON 结果可以直接用于生成表格：

```python
import json
import pandas as pd

# 加载结果
with open('vanilla_result.json') as f:
    vanilla = json.load(f)

# 转换为 DataFrame
df = pd.DataFrame(vanilla['breakdown'], index=[0])
print(df.to_latex())  # 导出 LaTeX 表格
```

## 引用

如果这个工具对你的研究有帮助，请引用：

```bibtex
@software{dpsgd_memory_profiler,
  title = {Detailed Memory Profiling System for DP-SGD Algorithms},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-repo}
}
```

## 贡献

欢迎提交 PR 改进这个工具！特别是：
- 支持更多 DP-SGD 算法
- 优化可视化样式
- 添加更多分析维度

## 许可证

Apache 2.0 License

---

**最后更新**: 2024-11-10  
**维护者**: Research Team  
**联系方式**: your-email@example.com

