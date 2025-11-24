FSDP延迟Norm计算死锁修复说明
=======================================

## 问题现象
在Ghost Clipping + FSDP模式下运行时出现死锁：
[Ghost] FSDP2 optimization enabled: disabling gradient sync for first backward
<程序挂起，无响应>

## 根本原因
Ghost Clipping使用两遍backward：
1. 第一遍：计算per-sample gradient norms（梯度同步被禁用）
2. 第二遍：应用clipping后的梯度（梯度同步启用）

在延迟模式下，hook在第一遍backward期间执行，如果此时调用
trainable_parameters(module)会触发FSDP all-gather操作，但由于
梯度同步被禁用，导致进程间通信死锁。

## 修复策略
将参数访问从hook执行期（unsafe）延迟到backward完成后（safe）

### 修复前（死锁）：
```
backward(retain_graph=True)        # sync禁用
  ├─ capture_backprops_hook         # hook执行
  │   ├─ trainable_parameters()     # ← 触发all-gather
  │   └─ DEADLOCK!                  # ← 死锁点
  └─ ...
```

### 修复后（正常）：
```
backward(retain_graph=True)        # sync禁用
  ├─ capture_backprops_hook         # hook执行
  │   ├─ 只缓存数据                 # ✓ 不访问参数
  │   └─ norm_sample = None         # ✓ 标记未初始化
  └─ backward完成
      ↓
set_requires_gradient_sync(True)    # 重新启用sync
      ↓
compute_all_norms_parallel()        # 现在访问参数是安全的
  ├─ trainable_parameters()         # ✓ sync已启用，不死锁
  └─ 初始化norm_sample并计算norms
```

## 代码修改位置
opacus/grad_sample/grad_sample_module_fast_gradient_clipping_fsdp.py:

1. capture_backprops_hook (line ~271-299):
   - 延迟模式下不调用trainable_parameters()
   - 只缓存activations、backprops、max_batch_len
   - 设置norm_sample = None

2. _compute_single_layer_norm (line ~644-689):
   - 在此处才调用trainable_parameters()（safe）
   - 此时backward已完成，梯度同步已恢复

## 验证方法
```bash
# 测试正确性（单GPU，无死锁）
cd memory_test/fsdp_llama3_profiling
python test_deferred_norm.py --mode correctness

# 测试性能（2 GPU FSDP，验证无死锁）
torchrun --nproc_per_node=2 test_deferred_norm.py --mode performance
```

预期：两个测试都应该正常完成，不出现挂起。

## 兼容性
- ✓ Flash Clipping（单遍backward）：无影响
- ✓ Ghost Clipping（两遍backward）：修复死锁
- ✓ Bookkeeping优化：兼容
- ✓ FSDP和单GPU模式：都支持
- ✓ 向后兼容：默认禁用延迟模式，不影响现有代码

## 关键学习
在FSDP环境下实现hook时的注意事项：
1. 了解FSDP的同步状态（enabled/disabled）
2. 避免在sync禁用时访问参数（会触发all-gather）
3. 将需要访问参数的操作延迟到sync恢复后
4. 在hook中只缓存最小必要信息

## 相关Issue
如果仍然遇到死锁：
1. 检查环境变量 OPACUS_USE_DEFERRED_NORM 是否设置为1
2. 确认使用的是修复后的代码版本
3. 查看日志确认"[Ghost] FSDP2 optimization enabled"消息后是否继续执行
4. 尝试禁用延迟模式：export OPACUS_USE_DEFERRED_NORM=0

