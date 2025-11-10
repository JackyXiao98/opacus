# 使用示例 (Usage Examples)

## 场景 1：完整实验套件（推荐）

这是最简单的方式，适合大多数用户：

```bash
cd /Users/bytedance/Desktop/Github/opacus
source .venv/bin/activate

# 运行完整的实验套件
./memory_test/test_algo/run_all_experiments.sh
```

**输出示例**：

```
========================================================================
Memory Profiling Experiment Suite
Each experiment runs in a fresh Python process to avoid contamination
========================================================================

Output directory: memory_profiling_results/run_20241110_202345

========================================================================
Running: vanilla
Output: memory_profiling_results/run_20241110_202345/vanilla_result.json
========================================================================

EXPERIMENT: Vanilla (No DP-SGD)
...
⏱️  Average iteration time: 7091.966 ms

DETAILED MEMORY BREAKDOWN
  Optimizer States                    10246.90 MB  ( 23.6%)
  Gradients                           15370.35 MB  ( 35.4%)
  Model Parameters                     5123.45 MB  ( 11.8%)
  ...
  Peak Allocated Mb                   43407.78 MB  (100.0%)

✓ vanilla completed

========================================================================
Running: ghost
...
========================================================================
Running: flash_clip
...
========================================================================
All experiments completed!
========================================================================

Generating visualizations...

✓ Saved: memory_profiling_results/run_20241110_202345/visualizations/memory_breakdown_comparison.png
✓ Saved: memory_profiling_results/run_20241110_202345/visualizations/memory_timeline.png
✓ Saved: memory_profiling_results/run_20241110_202345/visualizations/performance_tradeoff.png
✓ Saved: memory_profiling_results/run_20241110_202345/visualizations/summary.txt

✅ Complete pipeline finished!
✅ Results directory: memory_profiling_results/run_20241110_202345
✅ Visualizations: memory_profiling_results/run_20241110_202345/visualizations

========================================================================
SUMMARY
========================================================================
  vanilla: Peak Memory = 43407.78 MB, Avg Time = 7091.97 ms
  ghost: Peak Memory = 56992.12 MB, Avg Time = 17489.43 ms
  flash_clip: Peak Memory = 56816.12 MB, Avg Time = 7831.43 ms
========================================================================
```

---

## 场景 2：单独运行某个实验

如果只想测试某一个算法：

### Vanilla (基线)

```bash
python memory_test/test_algo/single_experiment.py \
    --experiment vanilla \
    --output results/vanilla.json \
    --vocab-size 32000 \
    --hidden-dim 2048 \
    --num-layers 20 \
    --num-heads 16 \
    --seq-len 16384 \
    --batch-size 1 \
    --num-iter 3 \
    --warmup-iter 2
```

### Ghost Clipping

```bash
python memory_test/test_algo/single_experiment.py \
    --experiment ghost \
    --output results/ghost.json \
    --seq-len 16384 \
    --batch-size 1
```

### Flash Clipping

```bash
python memory_test/test_algo/single_experiment.py \
    --experiment flash_clip \
    --output results/flash_clip.json \
    --seq-len 16384 \
    --batch-size 1
```

---

## 场景 3：小规模快速测试

如果想快速验证系统，可以使用更小的配置：

```bash
python memory_test/test_algo/single_experiment.py \
    --experiment vanilla \
    --output test_vanilla.json \
    --vocab-size 1000 \
    --hidden-dim 512 \
    --num-layers 4 \
    --num-heads 8 \
    --seq-len 1024 \
    --batch-size 2 \
    --num-iter 2 \
    --warmup-iter 1
```

这会在几分钟内完成，适合调试。

---

## 场景 4：自定义可视化

如果已经有了实验结果，想重新生成可视化：

```bash
python memory_test/test_algo/visualize_memory_breakdown.py \
    --input-dir memory_profiling_results/run_20241110_202345 \
    --output-dir my_custom_viz
```

---

## 场景 5：比较不同配置

比较不同序列长度的影响：

```bash
# 创建输出目录
mkdir -p results/seq_length_comparison

# 测试 seq_len = 8192
for exp in vanilla ghost flash_clip; do
    python memory_test/test_algo/single_experiment.py \
        --experiment $exp \
        --output results/seq_length_comparison/${exp}_8k.json \
        --seq-len 8192 \
        --batch-size 1
    sleep 5
done

# 测试 seq_len = 16384
for exp in vanilla ghost flash_clip; do
    python memory_test/test_algo/single_experiment.py \
        --experiment $exp \
        --output results/seq_length_comparison/${exp}_16k.json \
        --seq-len 16384 \
        --batch-size 1
    sleep 5
done

# 可以手动比较结果
python -c "
import json
for seq in ['8k', '16k']:
    for exp in ['vanilla', 'ghost', 'flash_clip']:
        with open(f'results/seq_length_comparison/{exp}_{seq}.json') as f:
            data = json.load(f)
        print(f'{exp} @ {seq}: {data[\"peak_memory_mb\"]:.2f} MB, {data[\"avg_time_ms\"]:.2f} ms')
"
```

---

## 场景 6：提取特定数据用于论文

### 提取峰值内存对比表

```python
import json
import pandas as pd

experiments = ['vanilla', 'ghost', 'flash_clip']
data = []

for exp in experiments:
    with open(f'results/{exp}.json') as f:
        result = json.load(f)
    
    data.append({
        'Method': exp.replace('_', ' ').title(),
        'Peak Memory (MB)': result['peak_memory_mb'],
        'Avg Time (ms)': result['avg_time_ms'],
        'Memory Overhead': result['peak_memory_mb'] - data[0]['Peak Memory (MB)'] if len(data) > 0 else 0
    })

df = pd.DataFrame(data)
print(df.to_markdown(index=False))
print("\n")
print(df.to_latex(index=False))
```

### 提取组件级分解

```python
import json

exp_name = 'ghost'
with open(f'results/{exp_name}.json') as f:
    result = json.load(f)

breakdown = result['breakdown']

print(f"\n{exp_name.upper()} Memory Breakdown:")
print("="*60)
for key, value in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
    if key.endswith('_mb') and value > 0:
        print(f"  {key.replace('_mb', '').replace('_', ' ').title():<40} {value:>10.2f} MB")
```

---

## 场景 7：持续监控（CI/CD 集成）

将这个系统集成到持续集成流程中：

```yaml
# .github/workflows/memory_profiling.yml
name: Memory Profiling

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  profile:
    runs-on: [self-hosted, gpu]  # 需要 GPU runner
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run memory profiling
      run: |
        ./memory_test/test_algo/run_all_experiments.sh
    
    - name: Upload results
      uses: actions/upload-artifact@v2
      with:
        name: profiling-results
        path: memory_profiling_results/
    
    - name: Check memory regression
      run: |
        python scripts/check_memory_regression.py \
          --baseline baseline_results.json \
          --current memory_profiling_results/run_*/ghost_result.json \
          --threshold 1.05  # 允许 5% 增长
```

---

## 场景 8：交互式分析

在 Jupyter Notebook 中交互式分析：

```python
# notebook.ipynb
import json
import matplotlib.pyplot as plt
import pandas as pd

# 加载结果
with open('memory_profiling_results/run_20241110_202345/ghost_result.json') as f:
    ghost = json.load(f)

# 查看时间线
snapshots = pd.DataFrame(ghost['snapshots'])
snapshots['stage'] = snapshots['name'].str.extract(r'(\d+_\w+)')

plt.figure(figsize=(14, 6))
plt.plot(snapshots.index, snapshots['allocated_mb'], 'o-', label='Allocated')
plt.plot(snapshots.index, snapshots['reserved_mb'], 's--', label='Reserved')
plt.xlabel('Timeline')
plt.ylabel('Memory (MB)')
plt.title('Ghost Clipping Memory Timeline')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(snapshots.index, snapshots['name'], rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 查看分解
breakdown = pd.Series(ghost['breakdown'])
breakdown = breakdown[breakdown.index.str.endswith('_mb')]
breakdown.plot(kind='bar', figsize=(10, 6))
plt.title('Ghost Clipping Memory Breakdown')
plt.ylabel('Memory (MB)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

---

## 常用命令速查

```bash
# 完整实验
./memory_test/test_algo/run_all_experiments.sh

# 单个实验（快速）
python memory_test/test_algo/single_experiment.py --experiment ghost --output ghost.json

# 测试系统
python memory_test/test_algo/test_profiler_system.py

# 重新生成可视化
python memory_test/test_algo/visualize_memory_breakdown.py \
    --input-dir memory_profiling_results/run_TIMESTAMP \
    --output-dir custom_viz

# 查看结果摘要
cat memory_profiling_results/run_TIMESTAMP/visualizations/summary.txt

# 比较两次运行
python -c "
import json
for exp in ['vanilla', 'ghost', 'flash_clip']:
    with open(f'memory_profiling_results/run_20241110_202345/{exp}_result.json') as f:
        data = json.load(f)
    print(f'{exp:15} {data[\"peak_memory_mb\"]:10.2f} MB  {data[\"avg_time_ms\"]:10.2f} ms')
"
```

---

## 故障排查速查

| 问题 | 解决方案 |
|------|---------|
| `CUDA out of memory` | 减小 `--seq-len` 或 `--batch-size` |
| `Permission denied` | `chmod +x run_all_experiments.sh` |
| `Module not found` | 确保在项目根目录运行，并激活虚拟环境 |
| 内存结果不准确 | 确保每个实验在独立进程中运行 |
| 可视化缺失 | 检查是否安装了 matplotlib: `pip install matplotlib` |

---

## 下一步

1. ✅ 运行完整实验套件
2. ✅ 查看生成的可视化
3. ✅ 阅读 summary.txt 了解详细分解
4. ✅ 根据需要调整配置并重新运行
5. ✅ 将结果用于论文或报告

**需要帮助？** 查看 `MEMORY_PROFILING_README.md` 获取更多信息。

