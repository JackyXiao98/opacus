延迟批量并行Norm计算优化 - 使用说明
================================================================

## 死锁修复
已修复延迟模式下的FSDP死锁问题（Ghost Clipping两遍backward）：

**根本原因：**
在Ghost Clipping模式下，第一次backward时FSDP梯度同步被禁用。
如果此时在hook中调用trainable_parameters()会触发FSDP all-gather，
但梯度同步被禁用，导致死锁。

**修复方案：**
1. Hook执行期间（backward中，sync禁用）：
   - 只缓存activations、backprops和max_batch_len
   - 不调用trainable_parameters()（避免触发all-gather）
   - 设置norm_sample=None标记为未初始化

2. Backward完成后（sync已重新启用）：
   - 在compute_all_norms_parallel()中才调用trainable_parameters()
   - 此时梯度同步已恢复，安全初始化norm_sample
   - 计算所有层的norms

**关键时机：**
```
backward(retain_graph=True)      # sync禁用，hook只缓存数据
  ↓
set_requires_gradient_sync(True)  # 重新启用sync
  ↓
compute_all_norms_parallel()      # 现在可以安全访问参数
```

## 使用方法

### 1. 启用延迟Norm计算

通过环境变量控制：
```bash
# 启用优化（延迟批量计算）
export OPACUS_USE_DEFERRED_NORM=1

# 禁用优化（即时计算，默认）
export OPACUS_USE_DEFERRED_NORM=0
```

### 2. 运行测试

测试文件位置：memory_test/fsdp_llama3_profiling/test_deferred_norm.py

#### 正确性测试（单GPU）
```bash
cd memory_test/fsdp_llama3_profiling
python test_deferred_norm.py --mode correctness
```

#### 性能测试（需要2个GPU）
```bash
cd memory_test/fsdp_llama3_profiling
torchrun --nproc_per_node=2 test_deferred_norm.py --mode performance
```

### 3. 在实际实验中使用

```bash
# 运行优化版本
OPACUS_USE_DEFERRED_NORM=1 python single_experiment.py \
  --mode flash_fsdp_bk \
  --seq-length 4096 \
  --batch-size 1 \
  --num-iter 3 \
  --output results/deferred.json \
  --model-name meta-llama/Llama-3.2-1B \
  --token YOUR_TOKEN

# 对比原始版本
OPACUS_USE_DEFERRED_NORM=0 python single_experiment.py \
  --mode flash_fsdp_bk \
  --seq-length 4096 \
  --batch-size 1 \
  --num-iter 3 \
  --output results/immediate.json \
  --model-name meta-llama/Llama-3.2-1B \
  --token YOUR_TOKEN
```

### 4. 启用详细profiling

```bash
export OPACUS_PROFILE_DETAILED=1
export OPACUS_USE_DEFERRED_NORM=1

python single_experiment.py --mode flash_fsdp_bk ...
```

预期输出：
```
[Deferred Norm] Computed 147 layer norms in parallel: 245.32 ms
[Deferred Norm] Average per layer: 1.67 ms

[Profile] Rank 0 First backward (norm pass): 1805.45 ms
[Profile] Rank 0 Parallel norm computation: 245.32 ms
[Profile] Rank 0 Get clipping coef: 3.21 ms
```

## 修改的文件

1. opacus/grad_sample/grad_sample_module_fast_gradient_clipping_fsdp.py
   - 添加_use_deferred_norm标志和_deferred_norm_cache
   - 修改capture_backprops_hook支持延迟收集
   - 新增compute_all_norms_parallel()批量计算方法
   - 修复FSDP死锁：避免在hook中访问参数

2. opacus/utils/fast_gradient_clipping_utils.py
   - 在backward()中调用compute_all_norms_parallel()

3. memory_test/fsdp_llama3_profiling/test_deferred_norm.py
   - 测试脚本，使用DP-compatible Flash Attention Transformer模型
   - 包含DPMultiheadAttentionWithFlashAttention（分离的Q/K/V投影）
   - 完全兼容Opacus DP-SGD要求

## 性能预期

在大模型（>1B参数）和长序列（>4096）上：
- 加速比: 1.5-2.0x
- 内存增长: < 5%

在小模型上加速效果可能不明显（测试用的SimpleTransformer）。

## 故障排除

### 1. 死锁问题
已修复。如果仍遇到死锁：
- 禁用优化：export OPACUS_USE_DEFERRED_NORM=0
- 检查FSDP配置是否正确
- 查看是否有其他hook干扰

### 2. 内存溢出
如果内存不足：
- 禁用优化：export OPACUS_USE_DEFERRED_NORM=0
- 减小batch size或sequence length

### 3. 正确性测试失败
如果误差过大：
- 检查是否使用了相同的随机种子
- 查看具体误差值（< 1e-4通常可接受）
- 确认两次运行使用相同的输入数据

## 兼容性

- ✅ 支持所有DP-SGD模式（flash, ghost, flash_bk, ghost_bk）
- ✅ 支持FSDP和单GPU模式
- ✅ 与Bookkeeping兼容
- ✅ 向后兼容，默认禁用不影响现有代码

