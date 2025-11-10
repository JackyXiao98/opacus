#!/usr/bin/env python3
"""
Quick test to verify the detailed memory profiling system works
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))

import torch
from detailed_memory_profiler import EnhancedMemoryProfiler, print_memory_breakdown
from memory_profile_with_flash_attention import SimpleBigModelWithFlashAttention


def test_basic_profiler():
    """Test basic profiler functionality"""
    print("\n" + "="*80)
    print("Testing Basic Profiler Functionality")
    print("="*80 + "\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cpu":
        print("⚠️  CUDA not available. Profiling on CPU (memory tracking disabled).")
    
    # Create small model
    config = {
        "vocab_size": 1000,
        "hidden_dim": 128,
        "num_layers": 2,
        "num_heads": 2,
        "seq_len": 256
    }
    
    print(f"Creating model with config: {config}")
    
    model = SimpleBigModelWithFlashAttention(
        vocab_size=config["vocab_size"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        seq_len=config["seq_len"]
    ).to(device)
    
    # Create profiler
    profiler = EnhancedMemoryProfiler(model, device)
    
    # Take snapshots
    profiler.take_snapshot("0_initial")
    print("✓ Initial snapshot taken")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    profiler.take_snapshot("1_optimizer_created")
    print("✓ Optimizer created")
    
    # Run one iteration
    batch_size = 2
    seq_len = 256
    
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len), device=device)
    labels = torch.randint(0, config["vocab_size"], (batch_size, seq_len), device=device)
    
    profiler.take_snapshot("2_before_forward")
    
    # Forward
    outputs = model(input_ids, labels=labels)
    loss = outputs["loss"]
    
    profiler.take_snapshot("3_after_forward")
    print("✓ Forward pass completed")
    
    # Backward
    loss.backward()
    
    profiler.take_snapshot("4_after_backward")
    print("✓ Backward pass completed")
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    profiler.take_snapshot("5_after_step")
    print("✓ Optimizer step completed")
    
    # Print breakdown
    breakdown = profiler.get_detailed_breakdown(optimizer)
    print_memory_breakdown(breakdown)
    
    # Print snapshots
    print("\n" + "="*80)
    print("MEMORY TIMELINE")
    print("="*80)
    for snapshot in profiler.snapshots:
        print(f"  {snapshot.name:<30} Allocated: {snapshot.allocated:>10.2f} MB  "
              f"Reserved: {snapshot.reserved:>10.2f} MB")
    print("="*80 + "\n")
    
    print("✅ Basic profiler test PASSED!")
    
    # Cleanup
    profiler.clear_hooks()
    del model, optimizer, profiler
    
    return True


def test_json_export():
    """Test JSON export functionality"""
    print("\n" + "="*80)
    print("Testing JSON Export")
    print("="*80 + "\n")
    
    import json
    import tempfile
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create small model
    model = SimpleBigModelWithFlashAttention(
        vocab_size=1000,
        hidden_dim=128,
        num_layers=2,
        num_heads=2,
        seq_len=256
    ).to(device)
    
    profiler = EnhancedMemoryProfiler(model, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    profiler.take_snapshot("test")
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    profiler.save_results(temp_path, optimizer)
    
    # Load and verify
    with open(temp_path, 'r') as f:
        data = json.load(f)
    
    assert "snapshots" in data, "Missing snapshots"
    assert "breakdown" in data, "Missing breakdown"
    assert len(data["snapshots"]) > 0, "No snapshots recorded"
    
    print(f"✓ JSON saved to: {temp_path}")
    print(f"✓ Snapshots: {len(data['snapshots'])}")
    print(f"✓ Breakdown keys: {list(data['breakdown'].keys())}")
    
    # Cleanup
    import os
    os.remove(temp_path)
    profiler.clear_hooks()
    del model, optimizer, profiler
    
    print("\n✅ JSON export test PASSED!")
    
    return True


def main():
    print("\n" + "="*80)
    print("DETAILED MEMORY PROFILER SYSTEM TEST")
    print("="*80)
    
    try:
        # Test 1: Basic functionality
        if not test_basic_profiler():
            print("❌ Basic profiler test FAILED")
            return False
        
        # Test 2: JSON export
        if not test_json_export():
            print("❌ JSON export test FAILED")
            return False
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nYou can now run the full experiment suite:")
        print("  ./memory_test/test_algo/run_all_experiments.sh")
        print("="*80 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

