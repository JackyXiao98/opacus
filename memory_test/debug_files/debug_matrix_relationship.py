import torch

def debug_matrix_relationship():
    """Debug the relationship between different matrix operations"""
    torch.manual_seed(42)
    
    # Test case
    B, T, d = 2, 4, 3
    G = torch.randn(B, T, d)
    
    print("=== Matrix Relationship Analysis ===")
    print(f"G shape: {G.shape}")
    
    # Compute GG^T
    ggT = torch.einsum("nik,njk->nij", G, G)  # [B, T, T]
    print(f"ggT shape: {ggT.shape}")
    
    # Different operations on ggT
    sum_all_elements = torch.sum(ggT, dim=(1, 2))
    trace_ggT = torch.einsum("nii->n", ggT)
    frobenius_ggT_squared = torch.einsum("nij,nij->n", ggT, ggT)
    
    print(f"Sum of all elements in ggT: {sum_all_elements}")
    print(f"Trace of ggT: {trace_ggT}")
    print(f"||ggT||_F^2: {frobenius_ggT_squared}")
    
    # Now let's understand what sum_all_elements actually represents
    # ggT[b, i, j] = sum_k G[b, i, k] * G[b, j, k]
    # sum_all_elements[b] = sum_{i,j} ggT[b, i, j] 
    #                     = sum_{i,j} sum_k G[b, i, k] * G[b, j, k]
    #                     = sum_k sum_{i,j} G[b, i, k] * G[b, j, k]
    #                     = sum_k (sum_i G[b, i, k]) * (sum_j G[b, j, k])
    #                     = sum_k (sum_i G[b, i, k])^2
    
    # Let's verify this
    sum_over_time = torch.sum(G, dim=1)  # [B, d] - sum over time dimension
    sum_squared = torch.sum(sum_over_time * sum_over_time, dim=1)  # [B]
    print(f"sum_k (sum_i G[b,i,k])^2: {sum_squared}")
    print(f"Are they equal? {torch.allclose(sum_all_elements, sum_squared)}")
    
    # Alternative computation
    sum_over_time_alt = torch.einsum("btd->bd", G)  # sum over time
    sum_squared_alt = torch.einsum("bd,bd->b", sum_over_time_alt, sum_over_time_alt)
    print(f"Alternative computation: {sum_squared_alt}")
    print(f"Are they equal? {torch.allclose(sum_all_elements, sum_squared_alt)}")
    
    # So the original algorithm computes:
    # sqrt(sum_k (sum_t G[b,t,k])^2)
    # This is the L2 norm of the sum over time dimension!
    
    print(f"\nOriginal algorithm computes: sqrt(||sum_t G[b,t,:]||_2^2)")
    print(f"This is NOT trace(ggT) and NOT ||G||_F")

if __name__ == "__main__":
    debug_matrix_relationship()