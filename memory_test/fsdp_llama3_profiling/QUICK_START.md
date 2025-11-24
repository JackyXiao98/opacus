延迟Norm计算优化 - 快速开始指南
=====================================

## 快速测试（验证死锁已修复）

### 1. 正确性测试（单GPU，2分钟）
```bash
cd memory_test/fsdp_llama3_profiling
python test_deferred_norm.py --mode correctness
```
预期输出：
```
✓ CORRECTNESS TEST PASSED!
```

### 2. 性能测试（需要2个GPU，5分钟）
```bash
cd memory_test/fsdp_llama3_profiling
torchrun --nproc_per_node=2 test_deferred_norm.py --mode performance
```
预期输出：
```
✓ PERFORMANCE TEST PASSED! (>1.2x speedup)
```

## 在真实模型上测试

### Llama3模型对比测试

#### 准备
```bash
export HF_TOKEN="your_huggingface_token"
cd memory_test/fsdp_llama3_profiling
mkdir -p results
```

#### 运行immediate模式（baseline）
```bash
OPACUS_USE_DEFERRED_NORM=0 \
python single_experiment.py \
  --mode flash_fsdp_bk \
  --seq-length 4096 \
  --batch-size 1 \
  --num-iter 3 \
  --output results/llama3_immediate_result.json \
  --model-name meta-llama/Llama-3.2-1B \
  --token $HF_TOKEN
```

#### 运行deferred模式（优化）
```bash
OPACUS_USE_DEFERRED_NORM=1 \
python single_experiment.py \
  --mode flash_fsdp_bk \
  --seq-length 4096 \
  --batch-size 1 \
  --num-iter 3 \
  --output results/llama3_deferred_result.json \
  --model-name meta-llama/Llama-3.2-1B \
  --token $HF_TOKEN
```

#### 对比结果
```bash
python -c "
import json
with open('results/llama3_immediate_result.json') as f:
    immediate = json.load(f)
with open('results/llama3_deferred_result.json') as f:
    deferred = json.load(f)

print('Results Comparison:')
print(f'Immediate: {immediate[\"avg_time_ms\"]:.2f} ms')
print(f'Deferred:  {deferred[\"avg_time_ms\"]:.2f} ms')
speedup = immediate['avg_time_ms'] / deferred['avg_time_ms']
print(f'Speedup:   {speedup:.2f}x')
"
```

## 预期性能

| 模型规模 | Seq Length | 预期加速比 |
|---------|------------|-----------|
| <500M   | 2048       | 1.2x      |
| <500M   | 4096       | 1.3x      |
| 1B-3B   | 4096       | 1.5-1.7x  |
| 1B-3B   | 8192       | 1.7-2.0x  |
| >7B     | 4096+      | 1.8-2.2x  |

注：小模型加速不明显，大模型和长序列效果更好

## 故障排除

### 问题1：测试挂起/死锁
**原因：** 可能使用了旧版本代码
**解决：** 
```bash
# 确认代码已更新
python -c "from opacus.grad_sample.grad_sample_module_fast_gradient_clipping_fsdp import GradSampleModuleFastGradientClippingFSDP; print('✓ Updated')"
```

### 问题2：正确性测试失败
**原因：** 随机种子不一致或数值精度问题
**解决：** 检查误差值，< 1e-4 通常可接受

### 问题3：性能没有提升
**原因：** 模型太小或batch size太小
**解决：** 尝试更大的模型或更长的序列

## 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| OPACUS_USE_DEFERRED_NORM | 0 | 1=启用优化, 0=禁用 |
| OPACUS_PROFILE_DETAILED | 0 | 1=详细profiling |
| OPACUS_PROFILE_FSDP | 0 | 1=FSDP profiling |
| OPACUS_PROFILE_FSDP_DETAILED | 0 | 1=FSDP详细profiling |

## 更多信息

- 详细使用说明: DEFERRED_NORM_USAGE.txt
- 死锁修复说明: DEADLOCK_FIX.txt
- 可视化结果: visualize_results.py

