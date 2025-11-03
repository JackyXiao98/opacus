Prompt for Claude 3 Sonnet / Claude 4.0 Sonnet:

你好，我需要你扮演一名精通 PyTorch、Opacus 和 NVIDIA Nsight Compute (ncu) 性能分析的专家。

我的目标是编写一个健壮的分析套件，用于深入分析不同训练算法（标准 SGD、DP-SGD、DP-SGD + Ghost Clipping）在训练大模型（约 1B 参数）时的 GPU 内核 (kernel) 性能，特别是 内存带宽/IO 开销 (I/O Cost)。

我们将使用 ncu 命令行工具来收集数据，并需要一个 Python 脚本来运行单一配置，以及一个 shell 脚本来自动执行所有实验。

请为我实现以下功能：

1. Python 脚本 (profiling_script_ncu.py)
这个脚本不应包含任何 torch.profiler 代码。它的 main 函数应该被参数化，以便从外部 shell 脚本接收单一实验配置。

Class 结构:

TrainerBase, StandardTrainer, DPSGDTrainer, DPGhostClippingTrainer: 保持与之前相似的结构（接收模型、优化器等）。

移除 run_profiling 方法。

将 profile_step 重命名为 train_step。

模型和数据:

SimpleBigModel: 可灵活调整参数（例如，用于 1B 的 GPU 模式和 1M 的 CPU 模式）。

get_random_dataloader: 可生成数据，并将其放到正确的 device 上。

run_local_test() 函数:

创建一个 run_local_test 函数，它在 'cpu' 设备上运行。

使用小模型（例如 1M 参数）快速遍历所有 TRAINER_CLASSES。

只调用一次 train_step 来确保代码在逻辑上是正确的（不崩溃）。

打印成功信息，例如 "[TEST OK] DPSGDTrainer passed local test."。

run_gpu_profile_step() 函数:

这是 ncu 将要分析的核心函数。

它接收参数: trainer_name, batch_size, seq_len。

它应在 'cuda' 设备上实例化大模型（约 1B 参数）。

它根据 trainer_name 实例化对应的 Trainer。

它获取一个 batch 的数据。

关键: 它应该运行几个步骤，例如 3 次预热 (warmup) 和 5 次激活 (active) 步骤，以便 ncu 能够稳定地捕获内核。

Python

# 示例:
# ... (初始化模型, trainer, dataloader)
# batch = next(iter(dataloader))
#
# print("Running warmup steps...")
# for _ in range(3):
#     trainer.train_step(batch)
#
# print("Running profiled steps...")
# # ncu 会自动捕获这些
# for _ in range(5):
#     trainer.train_step(batch)
#
# print("Profiled steps finished.")
if __name__ == "__main__": 块:

使用 argparse 解析命令行参数：

--mode (choices: 'test', 'profile', default: 'profile')

--trainer (choices: 'standard', 'dpsgd', 'dpsgd_ghost', default: 'standard')

--batch_size (type: int, default: 8)

--seq_len (type: int, default: 256)

如果 args.mode == 'test'，调用 run_local_test()。

如果 args.mode == 'profile'，调用 run_gpu_profile_step() 并传递其他 args。

2. Shell 启动脚本 (run_profiling_ncu.sh)
这是实验循环的控制中心。它将多次调用 ncu 和 Python 脚本。

脚本不得包含 set -e。

它应该定义实验变量：

Bash

TRAINERS=("standard" "dpsgd" "dpsgd_ghost")
BATCH_SIZES=(8 16)
SEQ_LENGTHS=(256 1024)
它应该使用嵌套循环遍历所有组合。

关键: 在循环内部，它应该构建并执行 ncu 命令，该命令专注于收集内存 I/O 指标：

Bash

#!/bin/bash

PYTHON_SCRIPT="profiling_script_ncu.py"

TRAINERS=("standard" "dpsgd" "dpsgd_ghost")
BATCH_SIZES=(8 16)
SEQ_LENGTHS=(256 1024)

# 关键的内存 I/O 指标
# 我们想知道 DRAM 和 L2 缓存的吞吐量
NCU_METRICS="dram_read_throughput,dram_write_throughput,l2_read_throughput,l2_write_throughput,gld_throughput,gst_throughput"

for trainer in "${TRAINERS[@]}"; do
  for bs in "${BATCH_SIZES[@]}"; do
    for seq in "${SEQ_LENGTHS[@]}"; do

      LOG_NAME="report_${trainer}_bs${bs}_seq${seq}"
      REPORT_FILE="${LOG_NAME}.ncu-rep"

      echo "-----------------------------------------------------"
      echo "Running: Trainer=${trainer}, BS=${bs}, SeqLen=${seq}"
      echo "Output file: ${REPORT_FILE}"
      echo "-----------------------------------------------------"

      # ncu 命令
      ncu -o "${REPORT_FILE}" \
          --metrics "${NCU_METRICS}" \
          --target-processes all \
          python "${PYTHON_SCRIPT}" \
              --mode=profile \
              --trainer="${trainer}" \
              --batch_size="${bs}" \
              --seq_len="${seq}"

      # 检查 ncu 是否成功 (可选, 但推荐)
      if [ $? -ne 0 ]; then
          echo "!!! NCU FAILED for ${LOG_NAME}"
      else
          echo "+++ NCU SUCCESS for ${LOG_NAME}"
      fi

    done
  done
done

echo "All profiling runs finished."
(请将上述 shell 脚本内容包含在你的交付成果中)

3. 交付成果
请提供两个文件：

profiling_script_ncu.py: 包含所有 Python 代码（Class、模型、run_local_test, run_gpu_profile_step, main）。

run_profiling_ncu.sh: 如上所述的、用于自动执行 ncu 的 shell 脚本。

最后，请在 profiling_script_ncu.py 脚本末尾的注释中，提供一个简短的ncu 分析指南，告诉我：

如何启动分析: "在 run_profiling_ncu.sh 运行完毕后，我将得到一堆 .ncu-rep 文件。我应该如何打开它们？（例如，使用 ncu-ui）"

如何分析 I/O Cost: "打开报告后 (例如 report_dpsgd_bs8_seq256.ncu-rep)，我应该在 GUI 的哪个视图中寻找什么？"

(最重要的) "为了证明 per-sample gradient clipping 带来了高昂的 I/O Cost，我应该具体查看哪些指标（例如，dram_read_throughput）？我应该如何比较 dpsgd 报告和 standard 报告以得出结论？"