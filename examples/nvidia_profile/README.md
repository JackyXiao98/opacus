# NVIDIA NCU Profiling for Opacus DP-SGD

这个目录包含用于分析 Opacus DP-SGD 性能的 NVIDIA Nsight Compute (NCU) 分析套件，专门用于量化 per-sample gradient clipping 带来的 GPU 内核性能和 I/O 开销。

## 文件说明

- `profiling_script_ncu.py`: 主要的 Python 分析脚本
- `run_profiling_ncu.sh`: 自动执行所有 NCU 实验的 Shell 脚本
- `prompt.md`: 详细的需求说明和指导文档
- `README.md`: 本说明文件

## 前置要求

### 软件依赖
1. **NVIDIA Nsight Compute**: 确保 `ncu` 命令在 PATH 中可用
2. **CUDA**: 支持的 CUDA 版本
3. **PyTorch**: 带 CUDA 支持的 PyTorch
4. **Opacus**: 最新版本的 Opacus 库

### 硬件要求
- NVIDIA GPU (推荐 V100, A100, 或更新的架构)
- 足够的 GPU 内存 (推荐 16GB+ 用于 1B 参数模型)

## 快速开始

### 1. 本地测试
首先运行本地测试确保所有组件正常工作：

```bash
cd examples/nvidia_profile
python profiling_script_ncu.py --mode=test
```

### 2. 单个实验
运行单个配置的 NCU 分析：

```bash
# 标准 SGD (基准)
ncu -o standard_test.ncu-rep --metrics dram_read_throughput,dram_write_throughput \
    python profiling_script_ncu.py --mode=profile --trainer=standard --batch_size=8 --seq_len=256

# DP-SGD (flat clipping)
ncu -o dpsgd_test.ncu-rep --metrics dram_read_throughput,dram_write_throughput \
    python profiling_script_ncu.py --mode=profile --trainer=dpsgd --batch_size=8 --seq_len=256
```

### 3. 完整实验套件
运行所有配置的自动化分析：

```bash
./run_profiling_ncu.sh
```

这将生成多个 `.ncu-rep` 文件在 `ncu_reports/` 目录中。

## 分析方法

### 1. 使用 NCU GUI
```bash
ncu-ui
```
在 GUI 中打开生成的 `.ncu-rep` 文件进行交互式分析。

### 2. 命令行分析
```bash
# 查看报告摘要
ncu --import ncu_reports/report_dpsgd_bs8_seq256.ncu-rep --page details

# 导出为 CSV 进行批量分析
ncu --import ncu_reports/report_dpsgd_bs8_seq256.ncu-rep --csv > analysis.csv
```

### 3. 关键指标对比

重点关注以下内存 I/O 指标来量化 per-sample gradient clipping 的开销：

| 指标 | 说明 | 预期差异 |
|------|------|----------|
| `dram_read_throughput` | DRAM 读取吞吐量 | DP-SGD > Standard (2-5x) |
| `dram_write_throughput` | DRAM 写入吞吐量 | DP-SGD > Standard (2-5x) |
| `l2_read_throughput` | L2 缓存读取吞吐量 | DP-SGD > Standard (1.5-3x) |
| `l2_write_throughput` | L2 缓存写入吞吐量 | DP-SGD > Standard (1.5-3x) |
| `gld_throughput` | 全局内存加载吞吐量 | DP-SGD > Standard |
| `gst_throughput` | 全局内存存储吞吐量 | DP-SGD > Standard |

### 4. 训练器对比

- **StandardTrainer**: 基准性能，最低的内存和计算开销
- **DPSGDTrainer**: 使用 flat clipping，显著增加的内存使用和 I/O 开销
- **DPGhostClippingTrainer**: 使用 Ghost Clipping，相比 flat clipping 减少内存使用

## 实验配置

脚本会测试以下配置组合：

- **训练器**: `standard`, `dpsgd`, `dpsgd_ghost`
- **批次大小**: `8`, `16`
- **序列长度**: `256`, `1024`

总共 3 × 2 × 2 = 12 个实验配置。

## 故障排除

### 常见问题

1. **NCU 未找到**
   ```
   ERROR: NCU (NVIDIA Nsight Compute) not found!
   ```
   解决方案: 安装 NVIDIA Nsight Compute 并确保 `ncu` 在 PATH 中。

2. **GPU 内存不足**
   ```
   CUDA out of memory
   ```
   解决方案: 减少 batch_size 或使用更小的模型配置。

3. **Opacus 不可用**
   ```
   Warning: Opacus not available
   ```
   解决方案: 安装 Opacus: `pip install opacus`

### 调试模式

如果遇到问题，可以先运行本地测试：

```bash
python profiling_script_ncu.py --mode=test
```

这将在 CPU 上使用小模型测试所有训练器的基本功能。

## 结果解读

### 预期的性能差异

1. **内存吞吐量**: DP-SGD 应该显示显著更高的 DRAM 读写吞吐量
2. **缓存压力**: L2 缓存吞吐量在 DP-SGD 中增加
3. **内核执行时间**: 梯度相关操作的执行时间增加
4. **Ghost Clipping 优化**: 相比 flat clipping 显示内存效率改进

### 分析报告

生成的 NCU 报告可以用于：
- 量化 per-sample gradient clipping 的性能开销
- 验证 Ghost Clipping 等优化技术的效果
- 识别性能瓶颈和优化机会
- 为不同硬件配置选择最佳的训练策略

## 参考资料

- [NVIDIA Nsight Compute 文档](https://docs.nvidia.com/nsight-compute/)
- [Opacus 官方文档](https://opacus.ai/)
- [PyTorch Profiler 指南](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)