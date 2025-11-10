# "Book-Keeping" (Ghost Clipping) 实现指南

本文档总结了在 PyTorch 中高效实现差分隐私随机梯度下降 (DP-SGD) 的 "Book-Keeping" 或 "幽灵裁剪" (Ghost Clipping) 技术的关键步骤。该技术的核心是通过在一次反向传播中完成单样本梯度范数的计算和裁剪，从而避免了传统实现中 `batch_size` 倍的计算开销。

## 核心思想

在标准的 `loss.backward()` 流程中，利用 PyTorch 的 `hooks` 机制，像“记账员”一样，在不中断主流程的情况下，完成以下任务：
1.  **前向传播**：记录计算梯度所需的中间变量（即层的输入激活）。
2.  **反向传播**：
    *   获取反向传播的梯度流。
    *   高效计算每个样本的梯度范数，而**不显式生成**完整的单样本梯度矩阵。
    *   根据范数计算裁剪因子。
    *   计算、裁剪并聚合梯度。

## 关键实现步骤

### 步骤 1: 挂载钩子 (Hooking)

遍历模型中所有需要进行隐私化训练的层 (例如 `nn.Linear`, `nn.Embedding`, `nn.Conv2d` 等)，并为它们注册前向和反向钩子。

```python
def add_hooks(model: nn.Module):
    for layer in model.modules():
        if isinstance(layer, nn.Linear): # 以 nn.Linear 为例
            layer.register_forward_hook(_save_activations_hook)
            layer.register_full_backward_hook(_compute_clipped_grad_hook) # 使用 full_backward_hook 更稳定
```

### 步骤 2: 前向钩子 - 保存激活 (Save Activations)

前向钩子的任务非常简单：捕获并保存该层的输入。这是计算梯度所必需的。

```python
def _save_activations_hook(layer, input, output):
    """前向钩子：保存输入激活。"""
    # input 是一个元组，通常我们关心第一个元素
    if not hasattr(layer, 'activations'):
        layer.activations = []
    layer.activations.append(input[0].detach())
```
**注意**: `detach()` 很重要，可以防止不必要的内存占用和计算图的增长。将激活保存在列表中可以处理模型中多次调用同一层的情况。

### 步骤 3: 反向钩子 - 计算、裁剪与聚合 (The Core Logic)

这是整个技术的核心。反向钩子在 `loss.backward()` 期间被触发，并接收到上游传来的梯度 (`grad_output`)。

```python
def _compute_clipped_grad_hook(layer, grad_input, grad_output):
    """
    反向钩子：执行幽灵裁剪的核心逻辑。
    """
    # 1. 获取所需张量
    A = layer.activations.pop(0)  # 来自前向钩子，遵循先进先出
    B = grad_output[0].detach() # 来自反向传播流

    # 2. 高效计算单样本梯度范数 (以线性层为例)
    # 这是关键优化：避免计算完整的 A^T @ B
    # 利用恒等式: ||A^T B||_F^2 = Tr(AA^T BB^T)
    # A 的形状: (batch_size, ..., in_features)
    # B 的形状: (batch_size, ..., out_features)
    
    # 为了处理序列等情况，先将中间维度展平
    A_flat = A.flatten(start_dim=1, end_dim=-2)
    B_flat = B.flatten(start_dim=1, end_dim=-2)

    # 使用 bmm (批量矩阵乘法) 高效计算
    # 注意：这里只计算了当前层权重的范数平方
    # 这是一个简化的例子，实际实现需要处理更复杂的维度
    per_sample_grad_norm_sq = (torch.bmm(A_flat.unsqueeze(2), A_flat.unsqueeze(1)) * \
                               torch.bmm(B_flat.unsqueeze(2), B_flat.unsqueeze(1))).sum(dim=(1,2))


    # 在实际应用中，需要一个全局的管理器来累加一个样本在所有层的范数
    # grad_norm_manager.add_norm(layer_name, per_sample_grad_norm_sq)

    # 3. (在所有层的钩子都执行完后) 计算裁剪因子
    # 假设 grad_norm_manager 已经收集了所有层的范数
    # total_norm_sq = grad_norm_manager.get_total_norm_sq_per_sample() # shape: (batch_size,)
    # total_norm = torch.sqrt(total_norm_sq)
    # clipping_factors = (max_grad_norm / (total_norm + 1e-6)).clamp(max=1.0) # shape: (batch_size,)

    # 4. 计算、裁剪并聚合梯度
    # 现在才真正计算梯度，并立即裁剪和聚合
    # einsum 是最灵活的方式
    # 需要将 clipping_factors 广播到正确的维度
    clipping_factors_expanded = clipping_factors.view(-1, 1, 1) 
    grad_sample = torch.einsum('bi,bo->bio', A, B)
    clipped_grad = torch.sum(grad_sample * clipping_factors_expanded, dim=0)

    # 5. 存储结果
    # 将计算好的梯度保存到参数的自定义属性中
    if hasattr(layer.weight, 'private_grad'):
        layer.weight.private_grad += clipped_grad.T # 注意转置
    else:
        layer.weight.private_grad = clipped_grad.T

    # 对偏置 (bias) 执行类似操作
    # ...
```

**重要说明**:
*   **范数聚合**: 上述代码片段简化了范数聚合的过程。在实际实现中，您需要一个上下文管理器或一个全局状态管理器，在反向传播期间收集并累加来自**所有**参与训练的层的单样本梯度范数平方，然后才能计算最终的裁剪因子。
*   **钩子执行顺序**: 反向钩子的执行顺序与层在反向图中的顺序一致。您需要确保在计算裁剪因子之前，所有相关层的钩子都已经执行完毕并记录了它们的范数。

### 步骤 4: 拦截优化器并更新参数

修改优化器的 `step` 方法，使其在更新参数前使用我们计算好的私有梯度。

```python
# 包装原始的 optimizer.step
original_step = optimizer.step

def private_step(closure=None):
    # 1. 添加噪声并赋值
    for p in model.parameters():
        if hasattr(p, 'private_grad'):
            noise = torch.normal(0, noise_multiplier * max_grad_norm, p.shape, device=p.device)
            # 将私有梯度赋值给标准梯度属性
            p.grad = p.private_grad / batch_size + noise
            del p.private_grad # 清理

    # 2. 调用原始更新规则
    original_step(closure)

optimizer.step = private_step
```

### 步骤 5: 清理

提供一个函数来移除所有添加的钩子，以便在推理或非私有训练时恢复模型。

```python
def remove_hooks(model: nn.Module):
    # 遍历所有钩子句柄 (handle) 并调用 handle.remove()
    for handle in hook_handles:
        handle.remove()
```

## 总结

通过以上步骤，可以在一次前向和一次反向传播中完成 DP-SGD 的核心逻辑，将训练开销从 `O(batch_size)` 降低到 `O(1)`，极大地提升了差分隐私深度学习的训练效率。关键在于**延迟梯度计算**和**高效的范数估算**。