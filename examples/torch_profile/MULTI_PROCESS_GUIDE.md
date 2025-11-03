# 多进程Profiling使用指南

## 🎯 概述

现在的profiling系统已经重构为多进程架构，每个实验配置都在独立的Python进程中运行，确保：
- ✅ 每个实验都有全新的Python环境
- ✅ 完全的内存隔离，避免内存累积
- ✅ 更好的错误隔离，单个实验失败不影响其他实验
- ✅ 详细的日志记录和结果汇总

## 📁 文件结构

```
torch_profile/
├── profiling_script.py          # 主要的profiling脚本（支持单配置运行）
├── run_profiling.sh             # Shell脚本（运行多个独立进程）
├── test_run.sh                  # 测试脚本（小配置验证）
├── summarize_results.py         # 结果汇总脚本
├── profiling_config.json        # 配置文件
├── logs/                        # 日志文件目录
├── runs/                        # TensorBoard结果目录
└── MULTI_PROCESS_GUIDE.md       # 本指南
```

## 🚀 使用方法

### 1. 运行完整的Profiling实验

```bash
# 运行所有配置（每个配置在独立进程中）
./run_profiling.sh
```

这将运行以下配置矩阵：
- **Trainers**: StandardTrainer, DPSGDTrainer, DPGhostClippingTrainer, DPFastGradientClippingTrainer
- **Batch sizes**: 4, 8
- **Sequence lengths**: 512, 1024
- **Model size**: 1B参数

### 2. 运行单个配置

```bash
# 运行单个配置
python profiling_script.py --mode=single \
    --trainer=StandardTrainer \
    --batch-size=4 \
    --seq-len=512 \
    --model-size=1b
```

### 3. 测试运行（小配置）

```bash
# 快速测试（使用小模型和小批次）
./test_run.sh
```

### 4. 查看结果汇总

```bash
# 生成汇总报告
python summarize_results.py --logs-dir=logs

# 保存详细结果到JSON文件
python summarize_results.py --logs-dir=logs --output=results.json
```

### 5. 查看TensorBoard结果

```bash
tensorboard --logdir=./runs
```

## ⚙️ 配置选项

### profiling_script.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--mode` | str | profile | 运行模式：profile/test/single |
| `--trainer` | str | - | 训练器类名（single模式必需）<br/>可选：StandardTrainer, DPSGDTrainer, DPGhostClippingTrainer, DPFastGradientClippingTrainer |
| `--batch-size` | int | 4 | 批次大小 |
| `--seq-len` | int | 512 | 序列长度 |
| `--model-size` | str | 1b | 模型大小：tiny/small/medium/1b |

### 模型大小配置

| 大小 | 参数量 | 词汇表 | 隐藏维度 | 层数 | 注意力头 |
|------|--------|--------|----------|------|----------|
| tiny | ~0.2M | 1,000 | 64 | 2 | 4 |
| small | ~8M | 8,000 | 256 | 4 | 8 |
| medium | ~100M | 16,000 | 512 | 8 | 8 |
| 1b | ~1.3B | 32,000 | 2,048 | 24 | 32 |

## 📊 输出说明

### 日志文件

每个配置的详细日志保存在 `logs/` 目录：
```
logs/
├── StandardTrainer_bs4_seq512.log
├── DPSGDTrainer_bs4_seq512.log
├── DPGhostClippingTrainer_bs4_seq512.log
└── ...
```

### TensorBoard文件

Profiling结果保存在 `runs/` 目录：
```
runs/
├── StandardTrainer_bs4_seq512/
├── DPSGDTrainer_bs4_seq512/
└── ...
```

### 汇总报告

`summarize_results.py` 生成的报告包含：
- 总体统计信息
- 按训练器分类的成功率
- 平均内存使用情况
- 失败实验的详细信息

## 🔧 自定义配置

### 修改run_profiling.sh

编辑脚本中的配置数组：
```bash
TRAINERS=("StandardTrainer" "DPSGDTrainer" "DPGhostClippingTrainer")
BATCH_SIZES=(4 8)
SEQ_LENGTHS=(512 1024)
MODEL_SIZE="1b"
```

### 使用配置文件

参考 `profiling_config.json` 来了解完整的配置选项。

## 🐛 故障排除

### 1. 内存不足

如果遇到GPU内存不足：
```bash
# 使用更小的批次大小
python profiling_script.py --mode=single --trainer=StandardTrainer --batch-size=2

# 或使用更小的模型
python profiling_script.py --mode=single --trainer=StandardTrainer --model-size=medium
```

### 2. 查看详细错误

检查具体的日志文件：
```bash
# 查看失败的实验日志
cat logs/DPGhostClippingTrainer_bs8_seq1024.log

# 搜索错误信息
grep -i "error\|exception\|failed" logs/*.log
```

### 3. 单独测试配置

在运行完整实验前，先测试单个配置：
```bash
python profiling_script.py --mode=single --trainer=DPGhostClippingTrainer --batch-size=4 --seq-len=512
```

## 💡 最佳实践

1. **先运行测试**: 使用 `./test_run.sh` 验证环境配置
2. **监控资源**: 运行时监控GPU内存和系统内存使用
3. **分批运行**: 对于大量配置，可以分批运行避免长时间占用资源
4. **保存结果**: 定期备份 `runs/` 和 `logs/` 目录
5. **清理缓存**: 实验间清理GPU缓存：`torch.cuda.empty_cache()`

## 🔄 与旧版本的区别

| 特性 | 旧版本 | 新版本 |
|------|--------|--------|
| 进程模式 | 单进程运行所有配置 | 每个配置独立进程 |
| 内存管理 | 手动清理，可能累积 | 自动隔离，无累积 |
| 错误处理 | 一个失败影响全部 | 错误隔离，独立处理 |
| 日志记录 | 混合在一起 | 每个配置独立日志 |
| 结果分析 | 手动分析 | 自动汇总报告 |

这种新的多进程架构确保了每个实验的独立性和可靠性，特别适合大规模的profiling实验。