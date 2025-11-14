#!/usr/bin/env python3
"""
Single GPU baseline training with Flash Clipping (non-FSDP).
This establishes the accuracy baseline for comparison with FSDP.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from opacus import PrivacyEngine
from test_model import SmallTransformerDP, create_synthetic_dataset


def train_single_gpu(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    num_epochs=1,
    save_dir="./results_single_gpu",
):
    """
    Train model on single GPU and track metrics.
    
    Returns:
        metrics: Dictionary containing training metrics
    """
    model = model.to(device)
    model.train()
    
    metrics = {
        "losses": [],
        "step_times": [],
        "grad_norms": [],
        "param_norms": [],
    }
    
    step = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            step_start = time.time()
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass (criterion.backward() for ghost/flash modes)
            loss.backward()
            
            # Track gradient norm (get from per-sample norms if available)
            total_grad_norm = 0.0
            if hasattr(model, 'per_sample_gradient_norms'):
                try:
                    # Get per-sample gradient norms from the module
                    per_sample_norms = model.per_sample_gradient_norms
                    # Average norm across the batch
                    total_grad_norm = float(per_sample_norms.mean().item())
                except:
                    total_grad_norm = 0.0
            
            optimizer.step()
            
            step_time = time.time() - step_start
            
            # Track parameter norm
            total_param_norm = 0.0
            for p in model.parameters():
                param_norm = p.data.norm(2)
                total_param_norm += param_norm.item() ** 2
            total_param_norm = total_param_norm ** 0.5
            
            # Record metrics
            metrics["losses"].append(loss.item())
            metrics["step_times"].append(step_time)
            metrics["grad_norms"].append(total_grad_norm)
            metrics["param_norms"].append(total_param_norm)
            
            epoch_loss += loss.item()
            step += 1
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Step {step}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Grad Norm: {total_grad_norm:.4f}")
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")
    
    # Save metrics
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save model checkpoint
    torch.save(model.state_dict(), os.path.join(save_dir, "model_checkpoint.pt"))
    
    print(f"Results saved to {save_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Single GPU Flash Clipping Training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--noise_multiplier", type=float, default=1.0, help="DP noise multiplier")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="./results_single_gpu", help="Save directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print("=" * 80)
    print("Single GPU Flash Clipping Training")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num samples: {args.num_samples}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Learning rate: {args.lr}")
    print(f"Noise multiplier: {args.noise_multiplier}")
    print(f"Max grad norm: {args.max_grad_norm}")
    print(f"Seed: {args.seed}")
    print("=" * 80)
    
    # Create model
    model = SmallTransformerDP(
        vocab_size=10000,
        hidden_dim=512,
        num_heads=8,
        num_layers=3,
        num_classes=10,
        max_seq_len=128,
        dropout=0.0,  # No dropout for deterministic results
    )
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Create dataset
    inputs, labels = create_synthetic_dataset(
        num_samples=args.num_samples,
        vocab_size=10000,
        seq_len=args.seq_len,
        num_classes=10,
        seed=args.seed,
    )
    dataset = TensorDataset(inputs, labels)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)  # No shuffle for reproducibility
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Initialize Privacy Engine
    privacy_engine = PrivacyEngine()
    model, optimizer, criterion, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        criterion=nn.CrossEntropyLoss(),
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
        grad_sample_mode="flash",  # Use flash clipping (non-FSDP)
        poisson_sampling=False,  # Disable for deterministic results
    )
    
    print(f"Privacy Engine initialized with grad_sample_mode='flash'")
    
    # Train
    metrics = train_single_gpu(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=args.device,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir,
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    print(f"Total steps: {len(metrics['losses'])}")
    print(f"Final loss: {metrics['losses'][-1]:.6f}")
    print(f"Average step time: {sum(metrics['step_times']) / len(metrics['step_times']):.4f}s")
    print(f"Final grad norm: {metrics['grad_norms'][-1]:.6f}")
    print(f"Final param norm: {metrics['param_norms'][-1]:.6f}")
    
    # Get privacy budget
    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    print(f"Privacy budget (ε): {epsilon:.2f} at δ=1e-5")
    print("=" * 80)


if __name__ == "__main__":
    main()

