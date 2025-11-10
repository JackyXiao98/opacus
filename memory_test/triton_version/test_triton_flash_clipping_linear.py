import torch
from triton_flash_clipping_linear import _triton_frobenius_inner_over_T, _triton_sum_over_time_norm_squared

def test_triton_functions():
    B, T, d_a, d_g = 2, 10, 16, 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.randn(B, T, d_a, device=device)
    G = torch.randn(B, T, d_g, device=device)

    # Test _triton_frobenius_inner_over_T
    torch_result_frobenius = 0
    for b in range(B):
        A_b = A[b]
        G_b = G[b]
        torch_result_frobenius += torch.sum(torch.matmul(G_b, G_b.T) * torch.matmul(A_b, A_b.T))

    triton_result_frobenius = _triton_frobenius_inner_over_T(A, G)
    
    # The torch_result_frobenius is a single value, but the triton_result_frobenius is a tensor of shape [B].
    # So we need to sum the triton_result_frobenius to get a single value.
    triton_result_frobenius_sum = torch.sum(triton_result_frobenius)

    # Test _triton_sum_over_time_norm_squared
    torch_result_sum_norm = torch.sum(torch.sum(G, dim=1)**2, dim=1)
    triton_result_sum_norm = _triton_sum_over_time_norm_squared(G)

    print(torch_result_frobenius)
    print(triton_result_frobenius_sum)
    print(torch_result_sum_norm)
    print(triton_result_sum_norm)

    assert torch.allclose(torch_result_frobenius, triton_result_frobenius_sum, atol=1e-4), "Frobenius inner product results do not match!"
    assert torch.allclose(torch_result_sum_norm, triton_result_sum_norm, atol=1e-4), "Sum over time norm squared results do not match!"

    print("All tests passed!")

if __name__ == "__main__":
    test_triton_functions()