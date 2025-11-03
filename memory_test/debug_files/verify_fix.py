import torch
import torch.nn as nn
from flash_clipping_linear import compute_linear_norm_sample as flash_compute
import sys
import os

# Add the opacus path to import the original implementation
sys.path.append('/Users/bytedance/Desktop/Github/opacus')
from opacus.grad_sample.linear import compute_linear_norm_sample as original_compute

def test_corrected_implementation():
    """Test the corrected flash implementation against the original"""
    torch.manual_seed(42)
    
    # Test parameters
    batch_size = 4
    seq_len = 16
    input_dim = 8
    output_dim = 6
    
    # Create a linear layer
    layer = nn.Linear(input_dim, output_dim, bias=True)
    
    # Test case 1: 2D input (standard case)
    print("=== Test Case 1: 2D Input ===")
    activations_2d = [torch.randn(batch_size, input_dim)]
    backprops_2d = torch.randn(batch_size, output_dim)
    
    original_result_2d = original_compute(layer, activations_2d, backprops_2d)
    flash_result_2d = flash_compute(layer, activations_2d, backprops_2d)
    
    print(f"Original weight norm: {original_result_2d[layer.weight]}")
    print(f"Flash weight norm: {flash_result_2d[layer.weight]}")
    print(f"Weight difference: {torch.abs(original_result_2d[layer.weight] - flash_result_2d[layer.weight])}")
    
    print(f"Original bias norm: {original_result_2d[layer.bias]}")
    print(f"Flash bias norm: {flash_result_2d[layer.bias]}")
    print(f"Bias difference: {torch.abs(original_result_2d[layer.bias] - flash_result_2d[layer.bias])}")
    
    # Test case 2: 3D input (sequence case - the problematic one)
    print("\n=== Test Case 2: 3D Input (Sequence) ===")
    activations_3d = [torch.randn(batch_size, seq_len, input_dim)]
    backprops_3d = torch.randn(batch_size, seq_len, output_dim)
    
    original_result_3d = original_compute(layer, activations_3d, backprops_3d)
    flash_result_3d = flash_compute(layer, activations_3d, backprops_3d)
    
    print(f"Original weight norm: {original_result_3d[layer.weight]}")
    print(f"Flash weight norm: {flash_result_3d[layer.weight]}")
    print(f"Weight difference: {torch.abs(original_result_3d[layer.weight] - flash_result_3d[layer.weight])}")
    print(f"Weight max relative error: {torch.max(torch.abs(original_result_3d[layer.weight] - flash_result_3d[layer.weight]) / original_result_3d[layer.weight])}")
    
    print(f"Original bias norm: {original_result_3d[layer.bias]}")
    print(f"Flash bias norm: {flash_result_3d[layer.bias]}")
    print(f"Bias difference: {torch.abs(original_result_3d[layer.bias] - flash_result_3d[layer.bias])}")
    print(f"Bias max relative error: {torch.max(torch.abs(original_result_3d[layer.bias] - flash_result_3d[layer.bias]) / original_result_3d[layer.bias])}")
    
    # Check if the results are close enough (within numerical precision)
    weight_close = torch.allclose(original_result_3d[layer.weight], flash_result_3d[layer.weight], rtol=1e-5, atol=1e-6)
    bias_close = torch.allclose(original_result_3d[layer.bias], flash_result_3d[layer.bias], rtol=1e-5, atol=1e-6)
    
    print(f"\nWeight results match (rtol=1e-5, atol=1e-6): {weight_close}")
    print(f"Bias results match (rtol=1e-5, atol=1e-6): {bias_close}")
    
    if weight_close and bias_close:
        print("\n✅ SUCCESS: Flash algorithm now matches the original implementation!")
    else:
        print("\n❌ FAILURE: There are still differences between implementations")
        
    return weight_close and bias_close

if __name__ == "__main__":
    success = test_corrected_implementation()
    exit(0 if success else 1)