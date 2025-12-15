#!/usr/bin/env python3
"""
Run a single profiling experiment with a Linear layer.
Supports single-GPU modes only.
This script is designed to be called from a shell script with different arguments.
"""

import argparse
import gc
import json
import os
import statistics

import torch
import torch.nn as nn
import torch.amp as torch_amp
from flashnorm import PrivacyEngine


# =============================================================================
# Simple Linear Model
# =============================================================================

class LinearModel(nn.Module):
    """
    Stacked Linear layer model for benchmarking.
    Input shape: [batch_size, seq_length, d]
    Output shape: [batch_size, seq_length, p]
    
    With num_layers > 1, creates a stack of Linear layers:
    - First layer: d -> p
    - Middle layers: p -> p
    - Last layer: p -> p (same as middle)
    """
    def __init__(self, d: int, p: int, num_layers: int = 1):
        super().__init__()
        self.num_layers = num_layers
        
        if num_layers == 1:
            self.layers = nn.ModuleList([nn.Linear(d, p)])
        else:
            layers = [nn.Linear(d, p)]  # First layer: d -> p
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(p, p))  # Subsequent layers: p -> p
            self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def count_parameters(self):
        return sum(param.numel() for param in self.parameters() if param.requires_grad)


class MSELossPerSample(nn.Module):
    """
    MSE loss that returns per-sample loss (reduced across feature dimension).
    
    This is needed for ghost clipping modes because:
    - Standard MSELoss with reduction='none' returns per-element loss [B*T, P]
    - Ghost clipping expects per-sample loss [B*T] or [B]
    - DPTensorFastGradientClipping.backward() does: torch.mean(loss_per_sample, dim=0)
    - If loss_per_sample is [B*T, P], result is [P], not a scalar -> RuntimeError
    
    This wrapper reduces across the feature dimension first to produce per-sample losses.
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, input, target):
        # Compute per-element MSE: [B*T, P]
        per_element_loss = torch.nn.functional.mse_loss(input, target, reduction='none')
        # Reduce across feature dimension to get per-sample loss: [B*T]
        per_sample_loss = per_element_loss.mean(dim=-1)
        
        if self.reduction == 'none':
            return per_sample_loss
        elif self.reduction == 'mean':
            return per_sample_loss.mean()
        elif self.reduction == 'sum':
            return per_sample_loss.sum()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


# =============================================================================
# Helper Functions
# =============================================================================

def aggressive_cleanup():
    """Aggressive memory cleanup"""
    for _ in range(3):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    gc.collect(generation=2)


def create_model(config, master_process=True, param_dtype=None):
    """Create Linear model"""
    d = config["d"]
    p = config["p"]
    num_layers = config.get("num_layers", 1)
    
    if master_process:
        print(f"Creating Linear model: d={d}, p={p}, num_layers={num_layers}")
    
    model = LinearModel(d=d, p=p, num_layers=num_layers)
    # if param_dtype is not None:
    #     model = model.to(dtype=param_dtype)
    
    if master_process:
        print(f"Model parameters: {model.count_parameters():,}")
    
    return model


def generate_batch(config, device):
    """Generate synthetic batch for linear layer"""
    batch_size = config["batch_size"]
    seq_length = config["seq_length"]
    d = config["d"]
    p = config["p"]
    
    # Input: [batch_size, seq_length, d]
    inputs = torch.randn(batch_size, seq_length, d, device=device)
    # Target: [batch_size, seq_length, p]
    targets = torch.randn(batch_size, seq_length, p, device=device)
    
    return {"inputs": inputs, "targets": targets}


# =============================================================================
# Experiment Runner
# =============================================================================

def run_experiment_worker(
    config,
    num_iter,
    warmup_iter,
    learning_rate,
    sigma,
    max_grad_norm,
):
    """Run experiment on a single GPU"""
    mode = config["mode"]
    time_stat = config.get("time_stat", "median")
    seq_length = config["seq_length"]
    batch_size = config["batch_size"]
    use_amp = config.get("use_amp", False)
    amp_dtype_str = config.get("amp_dtype", "bf16" if use_amp else None)
    if use_amp:
        amp_dtype = torch.bfloat16 if amp_dtype_str == "bf16" else torch.float16
    else:
        amp_dtype = None
    mp_label = f"{'bf16' if amp_dtype == torch.bfloat16 else 'fp16'}" if use_amp else "off"
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: mode={mode}, seq_length={seq_length}, batch_size={batch_size}")
    print(f"{'='*80}\n")
    
    device = torch.device("cuda:0")
    torch.manual_seed(1337)
    
    # Determine if this is a fuse mode (needs special handling)
    is_fuse_mode = mode in ["flash_fuse", "flash_fuse_bk"]
    
    # Create model (optionally initialize parameters in AMP dtype for fused path)
    init_dtype = amp_dtype if is_fuse_mode and use_amp and amp_dtype is not None else None
    model = create_model(config, param_dtype=init_dtype)
    
    # For fuse modes: Replace Linear with FusedFlashLinear
    if is_fuse_mode:
        print("Replacing Linear layers with FusedFlashLinear")
        from flashnorm.grad_sample.fused_flash_linear import replace_linear_with_fused
        model = replace_linear_with_fused(model)
    
    model = model.to(device)
    # Fused Flash kernel expects parameters to match activation dtype; keep weights
    # in AMP dtype when running fused modes to avoid dtype mismatch in backward.
    # if is_fuse_mode and use_amp and amp_dtype is not None:
    #     model = model.to(dtype=amp_dtype)
    model.train()
    
    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # Match transformer_experiment: use bf16 autocast without GradScaler; only
    # enable GradScaler for fp16
    scaler_enabled = use_amp and amp_dtype == torch.float16
    scaler = torch_amp.GradScaler("cuda", enabled=scaler_enabled)
    
    # Create criterion (MSE loss)
    # For ghost/ghost_bk and flash/flash_bk modes, use MSELossPerSample which returns
    # per-sample losses (standard MSELoss with reduction='none' returns per-element
    # losses [B*T, P], but DPLossFastGradientClipping expects per-sample losses [B*T])
    needs_per_sample_loss = mode in ["ghost", "ghost_bk", "flash", "flash_bk"]
    if needs_per_sample_loss:
        criterion = MSELossPerSample()
    else:
        criterion = nn.MSELoss()
    
    # Apply DP-SGD if needed (exclude no_dp_single)
    is_dp_mode = mode != "no_dp_single"
    
    if is_dp_mode:
        print(f"Applying DP-SGD with mode: {mode}")
        
        privacy_engine = PrivacyEngine()
        
        # Create dummy dataloader for make_private
        from torch.utils.data import TensorDataset, DataLoader
        
        dummy_data = torch.randn(batch_size, seq_length, config["d"])
        dummy_targets = torch.randn(batch_size, seq_length, config["p"])
        dummy_dataset = TensorDataset(dummy_data, dummy_targets)
        dummy_dataloader = DataLoader(
            dummy_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        
        # Map grad_materialize to "hooks" mode (standard Opacus)
        grad_sample_mode = "hooks" if mode == "grad_materialize" else mode
        
        # For hooks mode and fuse modes, don't pass criterion (they don't need wrapping)
        # Fuse modes handle gradient computation internally in FusedFlashLinear
        if grad_sample_mode == "hooks" or is_fuse_mode:
            model, optimizer, *_ = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=dummy_dataloader,
                noise_multiplier=sigma,
                max_grad_norm=max_grad_norm,
                grad_sample_mode=grad_sample_mode,
                poisson_sampling=False,
            )
        else:
            model, optimizer, criterion, *_ = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=dummy_dataloader,
                noise_multiplier=sigma,
                max_grad_norm=max_grad_norm,
                grad_sample_mode=grad_sample_mode,
                criterion=criterion,
                poisson_sampling=False,
            )
        
        print("DP-SGD setup complete")
    else:
        print("Running in non-DP mode")
    
    # Warmup iterations
    print(f"\nRunning {warmup_iter} warmup iterations... (AMP={mp_label})")
    
    for i in range(warmup_iter):
        batch = generate_batch(config, device)
        if use_amp and amp_dtype is not None:
            batch["inputs"] = batch["inputs"].to(dtype=amp_dtype)
            batch["targets"] = batch["targets"].to(dtype=amp_dtype)
        
        optimizer.zero_grad()
        with torch_amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            outputs = model(batch["inputs"])
            B, T, P = outputs.shape
            
            if needs_per_sample_loss:
                # Ghost/flash modes: criterion is wrapped with DPLossFastGradientClipping
                # Pass shape=(B, T, P) so it can reduce per-token losses [B*T] to per-sample [B]
                loss = criterion(outputs.view(B * T, P), batch["targets"].view(B * T, P), shape=(B, T, P))
            elif is_fuse_mode:
                # Fuse modes: keep 3D shape for proper FusedFlashLinear backward
                loss = criterion(outputs, batch["targets"])
            else:
                # Standard modes (no_dp_single, grad_materialize): standard loss computation
                loss = criterion(outputs.view(B * T, P), batch["targets"].view(B * T, P))
        
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        del batch, loss, outputs
        torch.cuda.empty_cache()
    
    # Reset memory stats after warmup
    torch.cuda.reset_peak_memory_stats(device)
    
    # Profiling iterations
    print(f"\nRunning {num_iter} profiling iterations... (AMP={mp_label})")
    
    iter_times = []
    
    for i in range(num_iter):
        batch = generate_batch(config, device)
        if use_amp and amp_dtype is not None:
            batch["inputs"] = batch["inputs"].to(dtype=amp_dtype)
            batch["targets"] = batch["targets"].to(dtype=amp_dtype)
        
        # Time the iteration
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        
        optimizer.zero_grad()
        
        with torch_amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            outputs = model(batch["inputs"])
            B, T, P = outputs.shape
            
            if needs_per_sample_loss:
                # Ghost/flash modes: criterion is wrapped with DPLossFastGradientClipping
                # Pass shape=(B, T, P) so it can reduce per-token losses [B*T] to per-sample [B]
                loss = criterion(outputs.view(B * T, P), batch["targets"].view(B * T, P), shape=(B, T, P))
            elif is_fuse_mode:
                # Fuse modes: keep 3D shape for proper FusedFlashLinear backward
                loss = criterion(outputs, batch["targets"])
            else:
                # Standard modes (no_dp_single, grad_materialize): standard loss computation
                loss = criterion(outputs.view(B * T, P), batch["targets"].view(B * T, P))
        
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        end_event.record()
        torch.cuda.synchronize()
        
        iter_time = start_event.elapsed_time(end_event)
        iter_times.append(iter_time)
        
        print(f"  Iteration {i+1}/{num_iter}: {iter_time:.2f} ms")
        
        del batch, loss, outputs
        torch.cuda.empty_cache()
    
    if time_stat == "mean":
        time_value_ms = sum(iter_times) / len(iter_times)
    elif time_stat == "median":
        time_value_ms = statistics.median(iter_times)
    else:
        raise ValueError(f"Unknown time_stat: {time_stat}")
    peak_memory_bytes = torch.cuda.max_memory_allocated(device)
    peak_memory_mb = peak_memory_bytes / (1024 ** 2)
    peak_memory_gb = peak_memory_bytes / (1024 ** 3)
    
    print(f"\n⏱️  Iteration time ({time_stat}): {time_value_ms:.2f} ms")
    print(f"💾 Peak memory usage: {peak_memory_gb:.2f} GB ({peak_memory_mb:.2f} MB)")
    
    return {
        "peak_memory_mb": peak_memory_mb,
        "peak_memory_gb": peak_memory_gb,
        "avg_time_ms": time_value_ms,
        "time_stat": time_stat,
    }


def run_experiment(config):
    """Run the experiment with the given configuration"""
    mode = config["mode"]
    seq_length = config["seq_length"]
    
    print(f"\n{'#'*80}")
    print(f"Configuration:")
    print(f"  Mode: {mode}")
    print(f"  d (input dim): {config['d']}")
    print(f"  p (output dim): {config['p']}")
    print(f"  Sequence Length: {seq_length}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Number of GPUs: 1 (single-GPU mode)")
    print(f"  Profiling Iterations: {config['num_iter']}")
    print(f"  Warmup Iterations: {config['warmup_iter']}")
    mp_cfg = config.get("amp_dtype") if config.get("use_amp") else None
    mp_desc = f"AMP {mp_cfg}" if mp_cfg else "None"
    print(f"  Mixed Precision: {mp_desc}")
    print(f"{'#'*80}\n")
    
    # Run experiment
    metrics = run_experiment_worker(
        config=config,
        num_iter=config["num_iter"],
        warmup_iter=config["warmup_iter"],
        learning_rate=config["learning_rate"],
        sigma=config["sigma"],
        max_grad_norm=config["max_grad_norm"],
    )
    
    # Build results
    results = {
        "mode": mode,
        "seq_length": seq_length,
        "batch_size": config["batch_size"],
        "total_batch_size": config["batch_size"],
        "num_gpus": 1,
        "d": config["d"],
        "p": config["p"],
        "num_layers": config.get("num_layers", 1),
        "peak_memory_mb": metrics["peak_memory_mb"],
        "peak_memory_gb": metrics["peak_memory_gb"],
        "avg_time_ms": metrics["avg_time_ms"],
        "time_stat": metrics["time_stat"],
        "config": config,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run single linear layer profiling experiment")
    
    # Mode arguments
    parser.add_argument("--mode", type=str, required=True,
                       choices=[
                           # Single-GPU modes
                           "no_dp_single", "grad_materialize",
                           "ghost", "ghost_bk", 
                           "flash", "flash_bk", 
                           "flash_fuse", "flash_fuse_bk"
                       ],
                       help="Training mode")
    
    # Model architecture arguments
    parser.add_argument("--d", type=int, default=512,
                       help="Input dimension of Linear layer")
    parser.add_argument("--p", type=int, default=512,
                       help="Output dimension of Linear layer")
    parser.add_argument("--num-layers", type=int, default=1,
                       help="Number of stacked Linear layers")
    parser.add_argument("--seq-length", type=int, default=1024,
                       help="Sequence length")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size")
    parser.add_argument("--num-iter", type=int, default=3,
                       help="Number of profiling iterations")
    parser.add_argument("--warmup-iter", type=int, default=1,
                       help="Number of warmup iterations")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--sigma", type=float, default=1.0,
                       help="Noise multiplier for DP")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                       help="Max gradient norm for DP")
    parser.add_argument("--mixed-precision", type=str, default="bf16",
                       choices=["none", "fp16", "bf16"],
                       help="Use CUDA AMP mixed precision (fp16/bf16)")
    parser.add_argument("--time-stat", type=str, default="median",
                       choices=["mean", "median"],
                       help="Statistic for iteration time reporting")
    
    # Output arguments
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSON file path")
    
    args = parser.parse_args()
    
    # Build config
    config = {
        "mode": args.mode,
        "d": args.d,
        "p": args.p,
        "num_layers": args.num_layers,
        "seq_length": args.seq_length,
        "batch_size": args.batch_size,
        "num_iter": args.num_iter,
        "warmup_iter": args.warmup_iter,
        "learning_rate": args.learning_rate,
        "sigma": args.sigma,
        "max_grad_norm": args.max_grad_norm,
        "use_amp": args.mixed_precision != "none",
        "amp_dtype": args.mixed_precision if args.mixed_precision != "none" else None,
        "time_stat": args.time_stat,
    }
    
    # Run experiment
    aggressive_cleanup()
    results = run_experiment(config)
    aggressive_cleanup()
    
    # Save results
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ Experiment completed!")
    print(f"✅ Mode: {results['mode']}")
    print(f"✅ Seq Length: {results['seq_length']}")
    print(f"✅ Peak Memory: {results['peak_memory_gb']:.2f} GB")
    print(f"✅ Time ({results['time_stat']}): {results['avg_time_ms']:.2f} ms")
    print(f"✅ Results saved to: {args.output}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

