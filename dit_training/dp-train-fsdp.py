# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP with Opacus FSDP support.
"""
import torch
import sys
import gc
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed._tensor.experimental import implicit_replication
from torch.distributed.tensor import Replicate, DTensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms


from opacus import PrivacyEngine
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.utils.fast_gradient_clipping_utils import DPTensorFastGradientClipping
from opacus.utils.fsdp_utils import FSDP2Wrapper
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
from logging import FileHandler  # 修正导入路径！

import os
from tqdm import tqdm, trange

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

# Add parent directories to path for DiT model imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    Handles DTensor parameters by operating on their local shards to avoid
    mixing torch.Tensor and DTensor in distributed operators.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    # Get the unwrapped model to access parameters correctly
    unwrapped_model = get_unwrapped_model(model)
    model_params = OrderedDict(unwrapped_model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        # Convert DTensor to replicated local tensor to match EMA parameter shape
        if isinstance(param, DTensor):
            from torch.distributed.tensor import Replicate, DTensor
            param_data = param.redistribute(placements=[Replicate()]).to_local()
        else:
            param_data = param.data

        if decay == 0:
            ema_params[name].data.copy_(param_data)
        else:
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)

# @torch.no_grad()
# def update_ema(ema_model, model, decay=0.9999):
#     """
#     支持FSDP/DTensor的EMA更新：本地张量运算+rank0广播，避免分布式通信爆炸
#     """
#     ema_params = OrderedDict(ema_model.named_parameters())
#     unwrapped_model = get_unwrapped_model(model)
#     model_params = OrderedDict(unwrapped_model.named_parameters())

#     # 分布式环境配置
#     is_distributed = dist.is_available() and dist.is_initialized()
#     rank = dist.get_rank() if is_distributed else 0
#     world_size = dist.get_world_size() if is_distributed else 1

#     for name, param in model_params.items():
#         if name not in ema_params:
#             continue

#         # 1. 将DTensor转为本地张量（关键：避免直接操作分布式张量）
#         if isinstance(param, torch.distributed.tensor.DTensor):
#             param_data = param.to_local()  # 获取当前进程的参数分片
#         else:
#             param_data = param.data

#         # 2. 初始化EMA（decay=0）：仅rank0赋值，然后广播
#         if decay == 0:
#             if rank == 0:
#                 ema_params[name].copy_(param_data)  # rank0用本地参数初始化
#             dist.broadcast(ema_params[name], src=0)  # 广播到所有进程
#             continue

#         # 3. 非初始化阶段：仅rank0收集所有分片（可选），更新后广播
#         full_param = param_data
#         if is_distributed and world_size > 1:
#             # 收集所有进程的分片（仅rank0需要）
#             gathered_params = [torch.zeros_like(param_data) for _ in range(world_size)] if rank == 0 else None
#             if rank == 0:
#                 dist.gather(param_data, gather_list=gathered_params, dst=0)
#                 full_param = torch.cat(gathered_params, dim=0)  # 拼接为完整参数

#         # 4. rank0更新EMA，然后广播
#         if rank == 0:
#             ema_params[name].mul_(decay).add_(full_param, alpha=1 - decay)
#         dist.broadcast(ema_params[name], src=0)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def get_unwrapped_model(model):
    """
    Get the underlying model from DDP/DPDDP/GradSampleModule/FSDP wrappers.
    
    After make_private() and FSDP wrapping, the model structure becomes:
    - GradSampleModule -> DPDDP -> FSDP2Wrapper -> original model
    
    This function unwraps all these layers to get the original model.
    """
    unwrapped = model
    while True:
        if hasattr(unwrapped, '_module'):  # Opacus GradSampleModule
            unwrapped = unwrapped._module
        elif hasattr(unwrapped, 'module'):  # DDP/DPDDP
            unwrapped = unwrapped.module
        else:
            break  # 已解包到原始模型
    return unwrapped


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Optimized version: Explicitly configures the module logger, does not depend on the root logger, and avoids being overridden by Opacus.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # 显式设置级别，不继承根logger
    logger.propagate = False  # 关键：禁止日志向上传播到根logger（避免被Opacus的根配置影响）

    if dist.get_rank() == 0:  # 仅rank=0输出日志
        # 控制台handler（带颜色）
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_formatter = logging.Formatter('[\033[34m%(asctime)s\033[0m] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        stream_handler.setFormatter(stream_formatter)

        # 文件handler
        file_handler = FileHandler(f"{logging_dir}/log.txt")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)

        # 给logger添加handler（不影响根logger）
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
    else:  # 其他rank添加NullHandler（不输出）
        logger.addHandler(logging.NullHandler())
    
    return logger
   
class CenterCropArr(torch.nn.Module):
    """
    可序列化的中心裁剪类，替代原有的lambda函数
    """
    def __init__(self, image_size):
        super().__init__()
        self.image_size = image_size  # 可通过self.parameters()管理（若有可学习参数）

    def forward(self, pil_image):
        # 复用原来的center_crop_arr逻辑
        while min(*pil_image.size) >= 2 * self.image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - self.image_size) // 2
        crop_x = (arr.shape[1] - self.image_size) // 2
        return Image.fromarray(arr[crop_y: crop_y + self.image_size, crop_x: crop_x + self.image_size])

    def __repr__(self):
        return f"CenterCropArr(image_size={self.image_size})"

def generate_dit_batch(config, device):
    """Generate synthetic batch for DiT (in latent space, same as dp-train.py)"""
    batch_size = config["batch_size"]
    in_channels = config["in_channels"]
    image_size = config["image_size"]
    num_classes = config["num_classes"]
    
    # DiT operates in latent space: latent_size = image_size // 8
    latent_size = image_size // 8
    
    # Generate latent representations (as if from VAE encoder)
    latents = torch.randn(batch_size, in_channels, latent_size, latent_size, device=device)
    timesteps = torch.randint(0, 1000, (batch_size,), device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    target_noise = torch.randn_like(latents)
    
    return {"images": latents, "timesteps": timesteps, "labels": labels, "target_noise": target_noise}


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(rank, world_size, args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    is_fsdp_mode = args.grad_sample_mode in ["ghost_fsdp", "flash_fsdp", "flash_fsdp_bk", "ghost_fsdp_bk", "flash_fsdp_fuse", "flash_fsdp_fuse_bk"]

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    assert args.global_batch_size % world_size == 0, f"Batch size must be divisible by world size."
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    if rank == 0:
        print("Create model! ", args.model)
        print(f"Model created. Device: {next(model.parameters()).device}")


    is_fuse_mode = args.grad_sample_mode in ["flash_fsdp_fuse", "flash_fsdp_fuse_bk", "flash_fuse", "flash_fuse_bk"]
    # For fuse modes: Replace Linear with FusedFlashLinear BEFORE FSDP wrapping
    if is_fuse_mode:
        if rank == 0:
            print("Replacing Linear layers with FusedFlashLinear (pre-FSDP)")
        from opacus.grad_sample.fused_flash_linear import replace_linear_with_fused
        model = replace_linear_with_fused(model)

    # Move model to device first
    model = model.to(device)

    # Create EMA model
    ema = deepcopy(model)
    requires_grad(ema, False)

    # Setup FSDP if needed
    if is_fsdp_mode:
        if rank == 0:
            print("Wrapping model with FSDP2")
        
        # Create DeviceMesh with named dimensions (required by FSDP2)
        from torch.distributed.device_mesh import init_device_mesh
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
        
        # Configure mixed precision
        if args.enable_mixed_precision:
            mp_policy = dist.fsdp.MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, 
                reduce_dtype=torch.float32
            )
        else:
            mp_policy = dist.fsdp.MixedPrecisionPolicy(
                param_dtype=torch.float32, 
                reduce_dtype=torch.float32
            )
        
        # Wrap model with FSDP2Wrapper
        model = FSDP2Wrapper(
            model, 
            mp_policy=mp_policy, 
            mesh=mesh, 
            use_block_wrapping=False, 
            wrap_individual_layers=False
        )
    else:
        if rank == 0:
            print("Using DDP mode (no FSDP)")
        # Use standard DDP for non-FSDP modes
        model = DDP(model, device_ids=[device])
    

    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in get_unwrapped_model(model).parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    if rank == 0:
        print("Setup data! ", args.data_path)
    transform = transforms.Compose([
        CenterCropArr(args.image_size),  # 替换lambda，使用可序列化的类
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    # For DP training with poisson_sampling=True:
    # - Do NOT use DistributedSampler (make_private handles distributed sampling)
    # - Use full batch_size (not divided by world_size)
    # For non-DP training:
    # - Use standard DDP approach with DistributedSampler and divided batch_size
    if args.dp_training:
        # DP training: non-distributed DataLoader, make_private will handle distribution
        loader = DataLoader(
            dataset,
            batch_size=args.global_batch_size,  # Full batch size
            shuffle=True,  # No DistributedSampler needed
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True
        )
        sampler = None  # No sampler for DP training
        logger.info(f"Dataset contains={len(dataset):,}, images=({args.data_path}), batch_size={args.global_batch_size} (will be distributed by make_private)")
    else:
        # Non-DP training: standard DDP with DistributedSampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.global_seed
        )
        loader = DataLoader(
            dataset,
            batch_size=int(args.global_batch_size // world_size),
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True
        )
        logger.info(f"Dataset contains={len(dataset):,}, images=({args.data_path}), batch_size={str(int(args.global_batch_size // world_size))}, total_gpus={str(world_size)}")

    def criterion(model, x, t, model_kwargs):
        return diffusion.training_losses(model, x, t, model_kwargs)['loss']
    
    # Set Privacy Training 
    if args.dp_training:
        logger.info(f"Starting DP Training!")
        privacy_engine = PrivacyEngine()
        model, opt, criterion, loader = privacy_engine.make_private(
            module=model,
            optimizer=opt,
            data_loader=loader,
            criterion=criterion,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
            grad_sample_mode=args.grad_sample_mode, 
            poisson_sampling=True,
        )
    else:
        logger.info(f"Starting Non-DP Training!")


    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in trange(args.epochs):
        # Only set_epoch for non-DP training (DP uses DPDataLoader with Poisson sampling)
        if sampler is not None:
            sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss = criterion(model, x, t, model_kwargs)
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            update_ema(ema, model)
            # torch.cuda.synchronize()
            
            # if is_fsdp_mode and dist.is_initialized():
            #     dist.barrier()

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    # Get unwrapped model for checkpointing
                    unwrapped_model = get_unwrapped_model(model)
                    checkpoint = {
                        "model": unwrapped_model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
            
            del loss
            torch.cuda.empty_cache()


    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image_size", type=int, choices=[256, 512, 1024], default=256)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=50000)

    # DP Training Setting
    parser.add_argument("--dp_training", type=bool, default=False)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--noise_multiplier", type=float, default=0.3701)
    parser.add_argument("--grad_sample_mode", type=str, default="flash_fsdp")
    
    # FSDP and mixed precision settings
    parser.add_argument("--enable_mixed_precision", type=bool, default=False)

    args = parser.parse_args()
    print(args)
    world_size = torch.cuda.device_count()

    mp.spawn(
        main,
        args=(world_size, args),
        nprocs=world_size,
        join=True,
    )