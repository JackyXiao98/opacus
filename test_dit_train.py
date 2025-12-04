# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms

from opacus import PrivacyEngine
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
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



#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def get_unwrapped_model(model):
    """
    Get the underlying model from DDP/DPDDP/GradSampleModule wrappers.
    
    After make_private(), the model structure becomes:
    - GradSampleModule -> DPDDP -> original model
    
    This function unwraps all these layers to get the original model.
    """
    # Unwrap GradSampleModule (uses _module attribute)
    if hasattr(model, '_module'):
        model = model._module
    # Unwrap DDP/DPDDP (uses module attribute)
    if hasattr(model, 'module'):
        model = model.module
    return model


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


# def create_logger(logging_dir):
#     """
#     Create a logger that writes to a log file and stdout.
#     """
#     if dist.get_rank() == 0:  # real logger
#         logging.basicConfig(
#             level=logging.INFO,
#             format='[\033[34m%(asctime)s\033[0m] %(message)s',
#             datefmt='%Y-%m-%d %H:%M:%S',
#             handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
#         )
#         logger = logging.getLogger(__name__)
#     else:  # dummy logger (does nothing)
#         logger = logging.getLogger(__name__)
#         logger.addHandler(logging.NullHandler())
#     return logger

def create_logger(logging_dir):
    """
    优化后：显式配置模块logger，不依赖根logger，避免被Opacus覆盖
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

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

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
        
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    
    # Wrap model with appropriate DDP wrapper based on training mode
    # For DP training: use DPDDP (Opacus's DifferentiallyPrivateDistributedDataParallel)
    # For non-DP training: use standard PyTorch DDP
    if args.dp_training:
        model = DPDDP(model.to(device))
    else:
        model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    if rank == 0:
        print("Setup data! ", args.data_path)
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
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
            drop_last=False
        )
        sampler = None  # No sampler for DP training
        logger.info(f"Dataset contains={len(dataset):,}, images=({args.data_path}), batch_size={args.global_batch_size} (will be distributed by make_private)")
    else:
        # Non-DP training: standard DDP with DistributedSampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=True,
            seed=args.global_seed
        )
        loader = DataLoader(
            dataset,
            batch_size=int(args.global_batch_size // dist.get_world_size()),
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        logger.info(f"Dataset contains={len(dataset):,}, images=({args.data_path}), batch_size={str(int(args.global_batch_size // dist.get_world_size()))}, total_gpus={str(dist.get_world_size())}")

    # Custom criterion for DiT that computes per-sample MSE loss
    # Required for flash_bk mode: must have .reduction attribute and return per-sample losses
    def dit_criterion(predicted, target):
        """
        Custom criterion for DiT that computes per-sample MSE loss.
        Matches the loss computation in diffusion.training_losses():
            terms["mse"] = mean_flat((target - model_output) ** 2)
        
        Args:
            predicted: (B, C, H, W) - model output (predicted noise/x_start/etc.)
            target: (B, C, H, W) - target (noise/x_start/etc. based on model_mean_type)
        Returns:
            loss_per_sample: (B,) - per-sample MSE loss
        """
        # Compute squared error and flatten all dims except batch, then take mean
        # This is equivalent to mean_flat((target - predicted) ** 2) in diffusion code
        return ((target - predicted) ** 2).flatten(start_dim=1).mean(dim=1)
    
    # Set reduction attribute (required by DPLossFastGradientClipping)
    dit_criterion.reduction = "mean"

    def compute_loss_non_dp(model, x, t, model_kwargs):
        """Compute diffusion training loss for non-DP training."""
        return diffusion.training_losses(model, x, t, model_kwargs)['loss']
    
    # Set Privacy Training 
    if args.dp_training:
        logger.info(f"Starting DP Training with flash_bk mode!")
        privacy_engine = PrivacyEngine()
        # Use "flash_bk" mode for memory efficiency with long sequences
        # flash_bk requires a criterion with .reduction attribute
        model, opt, dp_loss, loader = privacy_engine.make_private(
            module=model,
            optimizer=opt,
            data_loader=loader,
            criterion=dit_criterion,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
            grad_sample_mode="flash_bk",  # Flash Clipping + Bookkeeping for memory efficiency
            poisson_sampling=True,
        )
    else:
        logger.info(f"Starting Non-DP Training!")
        dp_loss = None  # Not used in non-DP training

    # Prepare models for training:
    update_ema(ema, get_unwrapped_model(model), decay=0)  # Ensure EMA is initialized with synced weights
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
            x = x.to(device)
            y = y.to(device)
            # print(x.shape, y.shape)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            
            if args.dp_training:
                # DP Training with flash_bk mode:
                # 1. Sample noise and add to latents using diffusion.q_sample
                # 2. Model forward pass to predict noise
                # 3. Compute loss via dp_loss(predicted, target)
                noise = torch.randn_like(x)
                noisy_x = diffusion.q_sample(x, t, noise=noise)  # x_t = sqrt(alpha_bar) * x + sqrt(1-alpha_bar) * noise
                
                # Forward pass: model predicts the noise
                predicted_noise = model(noisy_x, t, **model_kwargs)
                
                # Compute loss using dp_loss wrapper (handles flash clipping internally)
                loss = dp_loss(predicted_noise, noise)
            else:
                # Non-DP Training: use original diffusion loss computation
                loss = compute_loss_non_dp(model, x, t, model_kwargs)
            
            # print(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()

            update_ema(ema, get_unwrapped_model(model))

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
                avg_loss = avg_loss.item() / dist.get_world_size()
                # if rank == 0:
                #     print(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": get_unwrapped_model(model).state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

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
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=50000)

    # DP Training Setting
    parser.add_argument("--dp_training", type=bool, default=False)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--noise_multiplier", type=float, default=0.3701)


    args = parser.parse_args()
    print(args)
    main(args)