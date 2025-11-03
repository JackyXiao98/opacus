Prompt for Claude 3 Sonnet / Claude 4.0 Sonnet:

你好，我需要你扮演一名精通 PyTorch、Opacus 和 GPU 性能分析的专家。

我的目标是编写一个健壮的 Python 脚本，用于分析和比较不同训练算法（标准 SGD、DP-SGD、DP-SGD + Ghost Clipping）在训练大模型（约 1B 参数）时的 GPU 显存占用 (Memory Cost) 和 内存带宽/IO 开销 (I/O Cost)。

我希望使用 torch.profiler 和 TensorBoard 来进行可视化分析。

请为我实现以下功能：

1. 核心代码结构 (Class-Based)
请定义一个清晰的面向对象结构：

TrainerBase (抽象基类):

__init__ 中应接收 model, optimizer_cls, optimizer_params, device。

应包含一个 profile_step 方法，该方法接收 batch 数据并执行一个完整的训练步骤（forward, backward, step, zero_grad）。

应包含一个 run_profiling 方法，该方法设置 torch.profiler.profile 上下文管理器，并调用 profile_step。

StandardTrainer(TrainerBase):

__init__ 中设置一个标准的 torch.optim.SGD 优化器。

profile_step 执行标准的 loss.backward() 和 optimizer.step()。

DPSGDTrainer(TrainerBase):

__init__ 中设置 torch.optim.SGD。

关键: 它还应该初始化 opacus.PrivacyEngine，不使用 Ghost Clipping (clipping="flat")。

opacus.PrivacyEngine 将会 hook 模型和优化器。

DPGhostClippingTrainer(TrainerBase):

与 DPSGDTrainer 类似，但初始化 opacus.PrivacyEngine 时，应启用 Ghost Clipping (例如，clipping="per_layer" 或 clipping="adaptive")。 [请注意：Opacus 的 "Ghost Clipping" 通常是通过 clipping="per_layer" 或 clipping="adaptive" 来实现的，因为它避免了实例化完整的 per-sample gradient 矩阵。请使用 Opacus 推荐的最佳实践来实现这一点。]

2. 模型和数据
模型 (SimpleBigModel):

请定义一个简单的、可扩展的模型，例如一个大型 MLP 或一个简单的 Transformer 结构（例如，只有 nn.TransformerEncoderLayer 的堆叠）。

模型应接受参数（如 hidden_dim, num_layers, vocab_size），使其总参数量可以灵活调整（例如，用于 1B 的 GPU 模式和 1M 的 CPU 模式）。

请在代码中打印模型的总参数量以供确认。

模型应被移至 device (可以是 'cuda' 或 'cpu')。

数据 (get_random_dataloader):

编写一个函数，生成随机的 DataLoader。

输入: batch_size, seq_len, vocab_size, device。

输出: DataLoader，生成 (input_ids, labels) 的元组。

input_ids: torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

labels: torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

模型（如 Transformer）的 forward 方法应使用这些数据计算一个简单的 CrossEntropyLoss。

3. Profiling 和实验循环 (run_full_profile)
Profiler 配置: 在 run_profiling 方法中，torch.profiler.profile 应按如下方式配置：

Python

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1), # 1次等待, 2次预热, 3次激活
    on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./runs/{log_name}'),
    profile_memory=True,  # !!! 关键：开启显存分析
    record_shapes=True,
    with_stack=True
) as prof:
    # ... (循环调用 profile_step)
主实验函数 (run_full_profile):

请创建一个名为 run_full_profile 的函数，它在 'cuda' 设备上运行。

此函数包含主实验循环，迭代以下变量：

TRAINER_CLASSES = [StandardTrainer, DPSGDTrainer, DPGhostClippingTrainer]

BATCH_SIZES = [8, 16]

SEQ_LENGTHS = [256, 1024]

在此函数中，实例化的模型应为大模型（约 1B 参数）。

在每次循环开始时，重置模型和优化器（以避免干扰），并使用 torch.cuda.empty_cache()。

为每次实验组合生成一个唯一的 log_name（例如 f'DPSGD_bs{bs}_seq{seq}'），并传递给 tensorboard_trace_handler。

4. 本地单元测试 (run_local_test)
请创建第二个函数，名为 run_local_test，它在 'cpu' 设备上运行。

此函数用于快速测试代码可行性，不用于性能分析。

在此函数中，实例化的模型应为小模型（例如约 1M 参数）。

它应该遍历 TRAINER_CLASSES。

对于每个 Trainer，在 'cpu' 上实例化它，并使用小模型。

只调用一次 profile_step（不需要 torch.profiler 上下文），以确保代码能成功运行而不崩溃。

如果成功，打印例如 "[TEST OK] DPSGDTrainer passed local test." 的信息。

5. Shell 启动脚本 (run_profiling.sh)
请提供一个名为 run_profiling.sh 的 shell 脚本文件。

此脚本不得包含 set -e。如果某个组合出错，它应该继续尝试下一个（尽管在这个场景中我们只有一个命令）。

脚本内容应为：

Bash

#!/bin/bash
echo "Starting GPU profiling run..."
python profiling_script.py --mode=profile
echo "Profiling run finished."
(请假设 Python 脚本的文件名是 profiling_script.py)

6. 交付成果
请提供两个文件：

profiling_script.py:

包含所有 imports (torch, opacus, etc.)。

包含 Class 定义 (Base, Standard, DPSGD, Ghost)。

包含模型 (SimpleBigModel) 和数据 (get_random_dataloader) 的定义。

包含 run_full_profile (GPU, 1B模型) 和 run_local_test (CPU, 1M模型) 函数。

包含一个 if __name__ == "__main__": 块，它使用 argparse 来解析一个 --mode 参数 (默认为 'profile')。

如果 args.mode == 'profile'，则调用 run_full_profile()。

如果 args.mode == 'test'，则调用 run_local_test()。

run_profiling.sh: 如上所述的 shell 脚本。

最后，请在 profiling_script.py 脚本末尾的注释中，提供一个简短的分析指南，告诉我：

如何分析显存 (Memory Cost): "启动 tensorboard --logdir=./runs 后，我应该在 'PyTorch Profiler' 插件的哪个视图（例如 'Memory View'）中查看什么指标（例如 'Self CUDA Memory' 或 'Max Memory Allocated'），来对比 DP-SGD 和标准 SGD 之间巨大的显存差异？"

如何分析 I/O Cost: "在 'Trace Viewer' 或 'Kernel' 视图中，我应该寻找哪些特定的 CUDA Kernel 名称（例如，与 norm, clip, einsum 相关的），以及哪些指标（例如 'Duration' 或 'DRAM Read/Write'）来证明 per-sample gradient clipping 带来了高昂的计算或内存带宽（I/O）开销？"