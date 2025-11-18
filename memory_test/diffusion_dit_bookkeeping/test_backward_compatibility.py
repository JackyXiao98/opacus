#!/usr/bin/env python3
"""
Test backward compatibility: verify that standard single-input models
(like Transformers) still work correctly with the multi-input support changes.
"""

import torch
import torch.nn as nn
import sys
import os

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))

from opacus import GradSampleModule
from opacus.validators import ModuleValidator


class SimpleTransformerBlock(nn.Module):
    """
    A simple transformer block to test backward compatibility.
    This represents standard models like BERT/GPT that take single input.
    """
    def __init__(self, d_model=256, nhead=4, dim_feedforward=512):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        """Single input forward - standard Transformer"""
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)
        
        # FFN
        ff_output = self.linear2(self.activation(self.linear1(x)))
        x = self.norm2(x + ff_output)
        
        return x


class SimpleClassifier(nn.Module):
    """Simple MLP classifier for testing"""
    def __init__(self, input_dim=256, hidden_dim=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """Single input forward"""
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TransformerClassifier(nn.Module):
    """Complete Transformer-based classifier"""
    def __init__(self, seq_len=16, d_model=256, nhead=4, num_layers=2, num_classes=10):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        
        self.embedding = nn.Linear(d_model, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        self.transformer_blocks = nn.ModuleList([
            SimpleTransformerBlock(d_model, nhead)
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        """Single input: (B, seq_len, d_model)"""
        B, L, D = x.shape
        assert L == self.seq_len and D == self.d_model
        
        # Add positional embedding
        x = self.embedding(x) + self.pos_embedding
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Global average pooling + classification
        x = x.mean(dim=1)  # (B, d_model)
        x = self.classifier(x)
        
        return x


def test_simple_mlp():
    """Test that simple MLP still works with GradSampleModule"""
    print("\n" + "="*60)
    print("Test 1: Simple MLP Backward Compatibility")
    print("="*60)
    
    device = "cpu"  # Use CPU for testing
    batch_size = 4
    input_dim = 256
    num_classes = 10
    
    # Create model
    model = SimpleClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    model = ModuleValidator.fix(model)
    
    # Wrap with GradSampleModule
    gs_model = GradSampleModule(model, batch_first=True, loss_reduction="mean")
    
    # Create sample data
    x = torch.randn(batch_size, input_dim, device=device)
    target = torch.randint(0, num_classes, (batch_size,), device=device)
    
    # Forward pass
    gs_model.train()
    output = gs_model(x)
    assert output.shape == (batch_size, num_classes), f"Expected shape {(batch_size, num_classes)}, got {output.shape}"
    
    # Backward pass
    loss = nn.functional.cross_entropy(output, target)
    loss.backward()
    
    # Check grad_sample exists
    for name, param in gs_model.named_parameters():
        if param.requires_grad:
            assert hasattr(param, 'grad_sample'), f"Parameter {name} missing grad_sample"
            assert param.grad_sample is not None, f"Parameter {name} has None grad_sample"
            assert param.grad_sample.shape[0] == batch_size, \
                f"Parameter {name} grad_sample batch dimension is {param.grad_sample.shape[0]}, expected {batch_size}"
    
    print("✓ Simple MLP test passed!")
    print(f"  - Output shape: {output.shape}")
    print(f"  - All parameters have correct grad_sample shape")
    return True


def test_transformer_classifier():
    """Test that Transformer classifier works correctly"""
    print("\n" + "="*60)
    print("Test 2: Transformer Classifier Backward Compatibility")
    print("="*60)
    
    device = "cpu"
    batch_size = 4
    seq_len = 16
    d_model = 128  # Smaller for faster testing
    nhead = 4
    num_layers = 2
    num_classes = 10
    
    # Create model
    model = TransformerClassifier(
        seq_len=seq_len,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=num_classes
    ).to(device)
    model = ModuleValidator.fix(model)
    
    # Wrap with GradSampleModule
    gs_model = GradSampleModule(model, batch_first=True, loss_reduction="mean")
    
    # Create sample data
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    target = torch.randint(0, num_classes, (batch_size,), device=device)
    
    # Forward pass
    gs_model.train()
    output = gs_model(x)
    assert output.shape == (batch_size, num_classes), f"Expected shape {(batch_size, num_classes)}, got {output.shape}"
    
    # Backward pass
    loss = nn.functional.cross_entropy(output, target)
    loss.backward()
    
    # Check grad_sample exists and has correct shape
    grad_sample_count = 0
    for name, param in gs_model.named_parameters():
        if param.requires_grad:
            assert hasattr(param, 'grad_sample'), f"Parameter {name} missing grad_sample"
            assert param.grad_sample is not None, f"Parameter {name} has None grad_sample"
            assert param.grad_sample.shape[0] == batch_size, \
                f"Parameter {name} grad_sample batch dimension is {param.grad_sample.shape[0]}, expected {batch_size}"
            grad_sample_count += 1
    
    print("✓ Transformer classifier test passed!")
    print(f"  - Output shape: {output.shape}")
    print(f"  - {grad_sample_count} parameters have correct grad_sample shape")
    return True


def test_output_determinism():
    """Test that outputs are deterministic and gradients are correct"""
    print("\n" + "="*60)
    print("Test 3: Output Determinism and Gradient Correctness")
    print("="*60)
    
    device = "cpu"
    batch_size = 4
    input_dim = 128
    num_classes = 10
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    model1 = SimpleClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    model1 = ModuleValidator.fix(model1)
    
    torch.manual_seed(42)
    model2 = SimpleClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    model2 = ModuleValidator.fix(model2)
    
    # Wrap with GradSampleModule
    gs_model1 = GradSampleModule(model1, batch_first=True, loss_reduction="mean")
    gs_model2 = GradSampleModule(model2, batch_first=True, loss_reduction="mean")
    
    # Create sample data
    torch.manual_seed(123)
    x = torch.randn(batch_size, input_dim, device=device)
    target = torch.randint(0, num_classes, (batch_size,), device=device)
    
    # Forward pass 1
    gs_model1.train()
    output1 = gs_model1(x)
    loss1 = nn.functional.cross_entropy(output1, target)
    loss1.backward()
    
    # Forward pass 2
    gs_model2.train()
    output2 = gs_model2(x)
    loss2 = nn.functional.cross_entropy(output2, target)
    loss2.backward()
    
    # Check outputs are identical
    assert torch.allclose(output1, output2, atol=1e-6), "Outputs should be identical"
    assert torch.allclose(loss1, loss2, atol=1e-6), "Losses should be identical"
    
    # Check grad_samples are identical
    for (name1, param1), (name2, param2) in zip(gs_model1.named_parameters(), gs_model2.named_parameters()):
        assert name1 == name2
        if param1.requires_grad:
            assert torch.allclose(param1.grad_sample, param2.grad_sample, atol=1e-5), \
                f"Parameter {name1} grad_sample should be identical"
    
    print("✓ Determinism test passed!")
    print(f"  - Outputs are identical: max diff = {(output1 - output2).abs().max().item():.2e}")
    print(f"  - All grad_samples are identical")
    return True


def test_gradient_accumulation():
    """Test that gradient accumulation still works"""
    print("\n" + "="*60)
    print("Test 4: Gradient Accumulation")
    print("="*60)
    
    device = "cpu"
    batch_size = 4
    input_dim = 128
    num_classes = 10
    
    # Create model
    torch.manual_seed(42)
    model = SimpleClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    model = ModuleValidator.fix(model)
    gs_model = GradSampleModule(model, batch_first=True, loss_reduction="mean")
    
    # Allow gradient accumulation
    gs_model.allow_grad_accumulation()
    
    # First batch
    torch.manual_seed(100)
    x1 = torch.randn(batch_size, input_dim, device=device)
    target1 = torch.randint(0, num_classes, (batch_size,), device=device)
    
    gs_model.train()
    output1 = gs_model(x1)
    loss1 = nn.functional.cross_entropy(output1, target1)
    loss1.backward()
    
    # Check first batch grad_samples
    first_batch_grad_samples = {}
    for name, param in gs_model.named_parameters():
        if param.requires_grad and hasattr(param, 'grad_sample'):
            if isinstance(param.grad_sample, list):
                first_batch_grad_samples[name] = param.grad_sample[0].clone()
            else:
                first_batch_grad_samples[name] = param.grad_sample.clone()
    
    # Second batch (accumulation)
    torch.manual_seed(200)
    x2 = torch.randn(batch_size, input_dim, device=device)
    target2 = torch.randint(0, num_classes, (batch_size,), device=device)
    
    output2 = gs_model(x2)
    loss2 = nn.functional.cross_entropy(output2, target2)
    loss2.backward()
    
    # Check that grad_samples are accumulated (stored as list)
    for name, param in gs_model.named_parameters():
        if param.requires_grad and name in first_batch_grad_samples:
            assert isinstance(param.grad_sample, list), \
                f"Parameter {name} grad_sample should be a list after accumulation"
            assert len(param.grad_sample) == 2, \
                f"Parameter {name} grad_sample list should have 2 elements"
    
    print("✓ Gradient accumulation test passed!")
    print(f"  - grad_samples correctly accumulated as lists")
    return True


def test_per_sample_gradient_correctness():
    """Test that per-sample gradients are computed correctly"""
    print("\n" + "="*60)
    print("Test 5: Per-Sample Gradient Correctness")
    print("="*60)
    
    device = "cpu"
    batch_size = 3
    input_dim = 64
    hidden_dim = 32
    
    # Simple model for easier verification
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1)
    ).to(device)
    model = ModuleValidator.fix(model)
    
    # Create sample data
    torch.manual_seed(123)
    x = torch.randn(batch_size, input_dim, device=device)
    target = torch.randn(batch_size, 1, device=device)
    
    # Method 1: Compute per-sample gradients using GradSampleModule
    gs_model = GradSampleModule(model, batch_first=True, loss_reduction="mean")
    gs_model.train()
    output_batch = gs_model(x)
    loss_batch = nn.functional.mse_loss(output_batch, target)
    loss_batch.backward()
    
    # Get per-sample gradients
    per_sample_grads = {}
    for name, param in gs_model.named_parameters():
        if param.requires_grad and hasattr(param, 'grad_sample'):
            per_sample_grads[name] = param.grad_sample.clone()
    
    # Method 2: Compute gradients for each sample individually
    individual_grads = {name: [] for name in per_sample_grads.keys()}
    
    for i in range(batch_size):
        # Reset model
        torch.manual_seed(42)
        model_single = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)
        model_single = ModuleValidator.fix(model_single)
        model_single.train()
        
        # Single sample forward/backward
        x_single = x[i:i+1]
        target_single = target[i:i+1]
        output_single = model_single(x_single)
        loss_single = nn.functional.mse_loss(output_single, target_single)
        loss_single.backward()
        
        # Collect gradients - build name mapping
        for (name_batch, _), (name_single, param) in zip(gs_model.named_parameters(), model_single.named_parameters()):
            if param.requires_grad and param.grad is not None and name_batch in individual_grads:
                individual_grads[name_batch].append(param.grad.clone())
    
    # Compare: per-sample gradients should match individual gradients
    for name in per_sample_grads.keys():
        per_sample_grad_tensor = per_sample_grads[name]
        individual_grad_stacked = torch.stack(individual_grads[name], dim=0)
        
        # They should be close (allowing for numerical differences)
        max_diff = (per_sample_grad_tensor - individual_grad_stacked).abs().max().item()
        assert max_diff < 1e-4, \
            f"Parameter {name}: per-sample grads differ from individual grads by {max_diff:.2e}"
    
    print("✓ Per-sample gradient correctness test passed!")
    print(f"  - Per-sample gradients match individual sample gradients")
    print(f"  - Maximum difference across all parameters: {max_diff:.2e}")
    return True


def main():
    """Run all backward compatibility tests"""
    print("\n" + "="*70)
    print("BACKWARD COMPATIBILITY TEST SUITE")
    print("Testing that single-input models (Transformers) still work correctly")
    print("="*70)
    
    tests = [
        test_simple_mlp,
        test_transformer_classifier,
        test_output_determinism,
        test_gradient_accumulation,
        test_per_sample_gradient_correctness,
    ]
    
    results = []
    for test_fn in tests:
        try:
            result = test_fn()
            results.append((test_fn.__name__, True, None))
        except Exception as e:
            results.append((test_fn.__name__, False, str(e)))
            print(f"✗ {test_fn.__name__} FAILED: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, error in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {test_name}")
        if error:
            print(f"  Error: {error}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All backward compatibility tests passed!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())

