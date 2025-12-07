cd /mnt/bn/watermark/split_volume/zhaoyuchen/Project/diffusion-model-dp-training

# export NCCL_IB_DISABLE=0                  # 启用 IB 网络（若支持）
# export NCCL_IB_GID_INDEX=3                # IB 网络 GID 索引
# export NCCL_IB_HCA=  # RDMA 设备名称
# export NCCL_DEBUG=INFO                  # 调试时开启，查看 NCCL 通信日志
export NCCL_DEBUG=OFF        # 完全关闭 NCCL 调试输出
export NCCL_P2P_DISABLE=1    # 禁用 P2P 相关警告
export NCCL_SOCKET_IFNAME=eth0  # 指定网络接口（避免网络发现警告
export OMP_NUM_THREADS=2
export HF_HOME='/mnt/bn/watermark/split_volume/zhaoyuchen/Cache/huggingface'
export HF_ENDPOINT="https://hf-mirror.com"

export PATH=/mnt/bn/watermark/split_volume/zhaoyuchen/Environment/miniconda3/condabin:$PATH
if ! command -v conda &> /dev/null; then
    conda init bash
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dit

echo "training start"
echo "ARNOLD_ID:"$ARNOLD_ID
echo "ARNOLD_WORKER_0_HOST:"$ARNOLD_WORKER_0_HOST
echo "ARNOLD_WORKER_0_PORT:"$ARNOLD_WORKER_0_PORT
echo "MY_HOST_IP:":$MY_HOST_IP

MODEL=DiT-S
PATCH=2
IMAGE_SIZE=1024
EPOCH=450
NUM_CLASS=1
GLOBAL_BATCH_SIZE=64
NOISE_MULTIPLIER=0.9131
DATA_PATH=/mnt/bn/watermark/split_volume/zhaoyuchen/Dataset/celeba-hq/celeba_hq_onedir
RESULT_PATH=/mnt/bn/watermark/split_volume/zhaoyuchen/Dataset/dit-results/celebahq-non-dp-$MODEL-$PATCH-img$IMAGE_SIZE-cls$NUM_CLASS-bs$GLOBAL_BATCH_SIZE-epo$EPOCH
echo $RESULT_PATH

# export CUDA_LAUNCH_BLOCKING=1

export NCCL_NET_PLUGIN=none

nohup torchrun --nnodes=$ARNOLD_WORKER_NUM \
        --node_rank=$ARNOLD_ID \
        --master_addr=127.0.0.1 \
        --master_port=34534 \
        --nproc_per_node=8 \
        non-dp-train.py \
        --model $MODEL/$PATCH \
        --data_path $DATA_PATH \
        --epochs $EPOCH \
        --image_size $IMAGE_SIZE \
        --num_classes $NUM_CLASS \
        --results_dir $RESULT_PATH \
        --global_batch_size $GLOBAL_BATCH_SIZE \
        > ./output-non-dp-celebahq-training.log 2>&1 &
