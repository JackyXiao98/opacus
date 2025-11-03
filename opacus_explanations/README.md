# Opacus 解释文档集合

这个文件夹包含了对Opacus差分隐私库核心概念和实现的详细解释文档。

## 文件列表（按创建时间排序）

### 01_dpsgd_example.py
**DPSGD (Differentially Private Stochastic Gradient Descent) 最小可行示例**

- 📋 **内容**: 完整的DPSGD训练示例和核心概念演示
- 🎯 **重点**: 
  - Per-sample gradient计算演示
  - 梯度裁剪机制
  - 噪声添加过程
  - 完整DPSGD训练流程
- 🔧 **适用场景**: 初学者理解DPSGD基本原理

### 02_hook_explanation.py
**capture_backprops_hook 函数工作原理演示**

- 📋 **内容**: PyTorch hook机制在Opacus中的应用
- 🎯 **重点**:
  - Hook触发时机和工作流程
  - Per-sample gradient的捕获过程
  - 激活值和反向梯度的处理
  - 具体梯度计算演示
- 🔧 **适用场景**: 深入理解Opacus的hook实现机制

### 03_functorch_explanation.py
**ft_compute_sample_grad 函数实现原理演示**

- 📋 **内容**: functorch在Opacus中的动态函数创建机制
- 🎯 **重点**:
  - 动态函数创建过程
  - vmap机制的工作原理
  - functional模型转换
  - 手动循环 vs vmap的等价性演示
- 🔧 **适用场景**: 理解functorch的高级用法和优化原理

### 04_gradient_module_comparison.py
**GradSampleModule vs GradSampleModuleFastGradientClipping 详细对比**

- 📋 **内容**: 两种梯度采样模块的全面对比分析
- 🎯 **重点**:
  - 内存效率对比（99.9%的内存节省）
  - Ghost Clipping vs Fast Gradient Clipping
  - 工作流程差异
  - 适用场景分析
- 🔧 **适用场景**: 选择合适的梯度采样策略

## 使用方法

每个文件都可以独立运行：

```bash
cd opacus_explanations
python3 01_dpsgd_example.py
python3 02_hook_explanation.py
python3 03_functorch_explanation.py
python3 04_gradient_module_comparison.py
```

## 学习路径建议

1. **初学者**: 01 → 02 → 04 → 03
2. **进阶用户**: 04 → 03 → 02 → 01
3. **性能优化**: 04 → 03
4. **实现细节**: 02 → 03

## 核心概念总结

- **Per-sample Gradients**: 每个样本的独立梯度计算
- **Gradient Clipping**: 梯度裁剪保证隐私边界
- **Noise Addition**: 高斯噪声提供差分隐私保证
- **Hook Mechanism**: PyTorch钩子机制捕获梯度信息
- **Functorch Integration**: 函数式编程优化梯度计算
- **Memory Optimization**: Ghost Clipping等内存优化技术

## 相关资源

- [Opacus官方文档](https://opacus.ai/)
- [差分隐私理论](https://en.wikipedia.org/wiki/Differential_privacy)
- [PyTorch Functorch](https://pytorch.org/functorch/)

---

*这些解释文档旨在帮助理解Opacus的内部工作原理，适合研究人员和开发者深入学习差分隐私机器学习的实现细节。*