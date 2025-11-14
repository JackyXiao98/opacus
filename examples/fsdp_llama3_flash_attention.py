#!/usr/bin/env python3
import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import load_dataset
from huggingface_hub import login
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.utils.fsdp_utils import FSDP2Wrapper
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def prepare_snli_dataset(tokenizer, split="train", max_len=128):
    dataset = load_dataset("snli", split=split)
    dataset = dataset.filter(lambda example: example["label"] != -1)
    dataset = dataset.select(range(min(5000, len(dataset))))

    def tokenize_function(example):
        return tokenizer(
            example["premise"],
            example["hypothesis"],
            padding="max_length",
            max_length=max_len,
            truncation=True,
        )

    encoded_dataset = dataset.map(tokenize_function, batched=True)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return encoded_dataset


def _select_attn_impl(attn_impl: str):
    if attn_impl == "auto":
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except Exception:
            return "sdpa"
    if attn_impl == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except Exception:
            return "sdpa"
    return "sdpa"


def prepare_model(
    token: str,
    is_lora: bool = True,
    lora_rank: int = 16,
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    attn_impl: str = "auto",
):
    login(token)
    selected_impl = _select_attn_impl(attn_impl)
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3
    )
    pretrained_model.config.attn_implementation = selected_impl
    print(f"Attention implementation: {selected_impl}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        pretrained_model.config.pad_token_id = tokenizer.pad_token_id
    if is_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=32,
            lora_dropout=0.05,
        )
        model_with_lora = get_peft_model(pretrained_model, lora_config)

    trainable_parameters = 0
    if is_lora:
        for name, param in model_with_lora.named_parameters():
            if name == ("model.embed_tokens.weight"):
                param.requires_grad = False
            if param.requires_grad:
                trainable_parameters += param.numel()
    else:
        for name, param in pretrained_model.named_parameters():
            if name == ("model.embed_tokens.weight"):
                param.requires_grad = False
            if param.requires_grad:
                trainable_parameters += param.numel()

    print(f"Trainable parameters: {trainable_parameters}")
    if is_lora:
        return model_with_lora, tokenizer
    else:
        return pretrained_model, tokenizer


def train_step(model, optimizer, criterion, batch, device):
    optimizer.zero_grad()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["label"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    loss = criterion(outputs.logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss


def train(
    token: str,
    master_process: bool,
    rank: int,
    world_size: int,
    device: torch.device,
    is_lora: bool = True,
    lora_rank: int = 16,
    seq_length: int = 128,
    batch_size: int = 32,
    max_physical_batch_size: int = 1,
    learning_rate: float = 1e-5,
    sigma: float = 1,
    max_grad_norm: float = 1.0,
    epochs: int = 1,
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    mp_policy: dist.fsdp.MixedPrecisionPolicy = None,
    attn_impl: str = "auto",
):
    assert token is not None, "Please provide a valid huggingface token to access gated models"

    model_final, tokenizer = prepare_model(token, is_lora, lora_rank, model_name, attn_impl)
    train_dataset = prepare_snli_dataset(tokenizer, split="train", max_len=seq_length)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size // world_size,
        sampler=DistributedSampler(train_dataset),
    )

    model = FSDP2Wrapper(model_final, mp_policy=mp_policy)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    privacy_engine = PrivacyEngine()

    model, optimizer, criterion, train_dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_dataloader,
        noise_multiplier=sigma,
        max_grad_norm=max_grad_norm,
        grad_sample_mode="ghost_fsdp",
        criterion=torch.nn.CrossEntropyLoss(),
        poisson_sampling=False,
    )

    for epoch in range(1, epochs + 1):
        with BatchMemoryManager(
            data_loader=train_dataloader,
            max_physical_batch_size=max_physical_batch_size,
            optimizer=optimizer,
        ) as memory_safe_data_loader:
            for step, batch in tqdm(enumerate(memory_safe_data_loader), desc=f"Training epoch {epoch}: "):
                loss = train_step(model, optimizer, criterion, batch, device)
                if master_process:
                    print(f"Step: {step}, Loss: {loss.item()}")
    max_memory_allocated = torch.cuda.max_memory_allocated()
    print(f"Peak memory usage: {max_memory_allocated / 1024**3:.2f} GB on rank {rank}")


def launch(
    rank: int,
    world_size: int,
    token: str,
    batch_size: int = 32,
    max_physical_batch_size: int = 4,
    seq_length: int = 128,
    is_lora: bool = True,
    lora_rank: int = 8,
    learning_rate: float = 1e-5,
    sigma: float = 1.0,
    max_grad_norm: float = 1.0,
    epochs: int = 1,
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    mp_policy: dist.fsdp.MixedPrecisionPolicy = None,
    attn_impl: str = "auto",
):
    torch.cuda.set_device(rank)
    setup(rank, world_size)
    master_process = rank == 0
    seed_offset = rank

    tokens_per_iter = batch_size * seq_length
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs("/tmp/out", exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    train(
        token,
        master_process,
        rank,
        world_size,
        device=torch.device(f"cuda:{rank}"),
        seq_length=seq_length,
        batch_size=batch_size,
        max_physical_batch_size=max_physical_batch_size,
        lora_rank=lora_rank,
        is_lora=is_lora,
        learning_rate=learning_rate,
        sigma=sigma,
        max_grad_norm=max_grad_norm,
        epochs=epochs,
        model_name=model_name,
        mp_policy=mp_policy,
        attn_impl=attn_impl,
    )
    cleanup()


def main():
    parser = argparse.ArgumentParser(description="Distributed Llama Training Arguments (Flash Attention)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per process")
    parser.add_argument("--max_physical_batch_size", type=int, default=4, help="Max physical batch size")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length")
    parser.add_argument("--is_lora", type=bool, default=False, help="Use LoRA")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--sigma", type=float, default=1.0, help="Noise multiplier for DP")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max grad norm for DP")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B", help="Model name")
    parser.add_argument("--enable_mixed_precision", type=bool, default=True, help="enable mixed precision with bf16")
    parser.add_argument("--attn_impl", type=str, default="auto", choices=["auto", "sdpa", "flash_attention_2"], help="attention implementation")
    parser.add_argument("--token", type=str, default=None, help="Huggingface token", required=True)
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    print(f"Is cuda available: {torch.cuda.is_available()}, number of devices: {world_size}")
    if torch.cuda.current_device() == 0:
        print(f"Args: {args}")
    if args.enable_mixed_precision:
        mp_policy = dist.fsdp.MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    else:
        mp_policy = None

    selected_impl = _select_attn_impl(args.attn_impl)
    run_args = (
        world_size,
        args.token,
        args.batch_size,
        args.max_physical_batch_size,
        args.seq_length,
        args.is_lora,
        args.lora_rank,
        args.learning_rate,
        args.sigma,
        args.max_grad_norm,
        args.epochs,
        args.model_name,
        mp_policy,
        selected_impl,
    )
    mp.spawn(launch, args=run_args, nprocs=world_size, join=True)


if __name__ == "__main__":
    main()