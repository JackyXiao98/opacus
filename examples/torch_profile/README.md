# PyTorch Profiling for Opacus DP-SGD

这个目录包含用于分析和比较不同训练算法（标准 SGD、DP-SGD、DP-SGD + Ghost Clipping）在训练大模型时的 GPU 显存占用和内存带宽/IO 开销的工具。

## 文件说明

- `profiling_script.py`: 主要的性能分析脚本
- `run_profiling.sh`: 启动 GPU 性能分析的 shell 脚本
- `instruction.md`: 详细的实现指南和需求说明

## 依赖要求

### 方法 1: 使用 requirements.txt (推荐)

```bash
pip install -r requirements.txt
```

### 方法 2: 手动安装

```bash
pip install torch>=1.12.0 torchvision>=0.13.0
pip install opacus>=1.4.0
pip install tensorboard>=2.8.0
pip install numpy>=1.21.0
```

### 方法 3: 使用 conda (如果可用)

```bash
conda install pytorch torchvision -c pytorch
pip install opacus tensorboard
```

## 使用方法

### 1. 本地测试（CPU，小模型）

```bash
python3 profiling_script.py --mode=test
```

这将在 CPU 上使用小模型（约 1M 参数）快速测试所有训练器的功能。

### 2. 完整性能分析（GPU，大模型）

```bash
# 方法 1: 直接运行
python3 profiling_script.py --mode=profile

# 方法 2: 使用 shell 脚本
./run_profiling.sh
```

这将在 GPU 上使用大模型（约 1B 参数）进行完整的性能分析。

### 3. 查看结果

```bash
tensorboard --logdir=./runs
```

然后在浏览器中打开 http://localhost:6006 查看 TensorBoard 界面。

## 训练器类型

1. **StandardTrainer**: 标准 SGD 训练，无差分隐私
2. **DPSGDTrainer**: DP-SGD 训练，使用 `GradSampleModule` 和 `DPOptimizer`（标准 per-sample 梯度）
3. **DPGhostClippingTrainer**: DP-SGD 训练，使用 Ghost Clipping（`wrap_model` 与 `grad_sample_mode="ghost"`，内存高效）

## Opacus 兼容性

本实现已针对 Opacus 进行了优化：

- **模型架构**: 使用 `DPMultiheadAttention` 替代标准的 `nn.MultiheadAttention`
- **DP 训练**: 使用 `GradSampleModule` 和 `DPOptimizer` 的现代 API
- **Ghost Clipping**: 通过 `wrap_model` 函数和 `grad_sample_mode="ghost"` 实现内存高效的梯度裁剪

## 实验配置

- **批次大小**: [4, 8] (GPU) 或 [2, 4] (CPU)
- **序列长度**: [512, 1024] (GPU) 或 [256, 512] (CPU)
- **模型大小**: 约 1B 参数 (GPU) 或 1M 参数 (CPU)

## 分析指南

### 显存分析 (Memory Cost)

在 TensorBoard 的 'PyTorch Profiler' 插件中：

1. 进入 'Memory View' 标签页
2. 查看关键指标：
   - 'Self CUDA Memory': 每个操作的直接显存使用量
   - 'CUDA Memory Usage': 总体显存使用趋势图
   - 'Memory Timeline': 显存分配和释放的时间线

### I/O 开销分析 (I/O Cost)

在 'Trace Viewer' 或 'Kernel' 视图中：

1. 查找关键的 CUDA Kernel：
   - 'elementwise_kernel': 逐元素操作
   - 'reduce_kernel': 归约操作
   - 包含 'clip', 'norm', 'einsum' 的 kernel

2. 关注性能指标：
   - 'Duration': kernel 执行时间
   - GPU 利用率和内存带宽使用

## 预期结果

- **Standard SGD**: 最低的显存和计算开销
- **DP-SGD (flat clipping)**: 显著增加的显存使用和 I/O 开销
- **DP-SGD (Ghost Clipping)**: 相比 flat clipping 减少的显存使用

## 安装验证

在运行主脚本之前，可以使用语法检查工具验证代码结构：

```bash
python3 test_syntax.py
```

## 故障排除

1. **PyTorch 未安装**: 
   ```
   ModuleNotFoundError: No module named 'torch'
   ```
   解决方案：按照上述依赖要求安装 PyTorch

2. **Opacus 未安装**: 
   ```
   Warning: Opacus not available. DP-SGD trainers will not work.
   ```
   解决方案：`pip install opacus>=1.4.0`

3. **CUDA 不可用**: 脚本会自动切换到 CPU 模式

4. **内存不足**: 尝试减少批次大小或序列长度

5. **NoneType 错误**: 
   ```
   object of type 'NoneType' has no len()
   ```
   这个问题已在最新版本中修复，确保使用 `GradSampleModule` 和 `DPOptimizer`

6. **MultiheadAttention 兼容性错误**:
   ```
   ShouldReplaceModuleError: We do not support nn.MultiheadAttention
   ```
   这个问题已修复，现在使用 `DPMultiheadAttention` 和兼容的模型架构

7. **GradSampleModule 参数错误**:
   ```
   GradSampleModule.__init__() got an unexpected keyword argument 'grad_sample_mode'
   ```
   这个问题已修复，现在使用 `wrap_model` 函数来正确设置 Ghost Clipping

8. **张量 view 兼容性错误**:
   ```
   view size is not compatible with input tensor's size and stride
   ```
   这个问题已修复，现在使用 `.reshape()` 替代 `.view()` 来处理非连续张量

9. **Opacus 内部张量兼容性错误**:
   ```
   RuntimeError: view size is not compatible with input tensor's size and stride
   (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
   ```
   这个问题已修复，现在确保所有输入张量在传递给 Opacus 前都是连续的（使用 `.contiguous()`）

## 注意事项

- 确保有足够的 GPU 显存（建议至少 8GB）
- 首次运行可能需要下载和编译 CUDA kernels
- 性能分析会产生大量数据，确保有足够的磁盘空间