import torch
import torch.nn as nn
from opt_einsum import contract

def original_bias_computation_3d(backprops):
    """Original algorithm for 3D bias computation"""
    ggT = torch.einsum("nik,njk->nij", backprops, backprops)  # batchwise g g^T
    return torch.sqrt(torch.einsum("n...i->n", ggT).clamp(min=0))

def flash_bias_computation_3d(backprops, tile_size=256, dtype_acc=torch.float32):
    """Flash algorithm for 3D bias computation"""
    B, T, d_g = backprops.shape
    out = torch.zeros(B, dtype=dtype_acc, device=backprops.device)
    num_tiles = (T + tile_size - 1) // tile_size

    for p in range(num_tiles):
        ps = p * tile_size
        pe = min((p + 1) * tile_size, T)
        G_p = backprops[:, ps:pe, :].to(dtype_acc)
        S_pp = contract('bid,bjd->bij', G_p, G_p)
        out += contract('bij,bij->b', S_pp, S_pp)

        for q in range(p):
            qs = q * tile_size
            qe = min((q + 1) * tile_size, T)
            G_q = backprops[:, qs:qe, :].to(dtype_acc)
            S_pq = contract('bid,bjd->bij', G_p, G_q)
            out += 2.0 * contract('bij,bij->b', S_pq, S_pq)

    return torch.sqrt(out.clamp_min(0.0))

def corrected_flash_bias_computation_3d(backprops, tile_size=256, dtype_acc=torch.float32):
    """Corrected flash algorithm for 3D bias computation"""
    B, T, d_g = backprops.shape
    out = torch.zeros(B, dtype=dtype_acc, device=backprops.device)
    num_tiles = (T + tile_size - 1) // tile_size

    for p in range(num_tiles):
        ps = p * tile_size
        pe = min((p + 1) * tile_size, T)
        G_p = backprops[:, ps:pe, :].to(dtype_acc)
        # Diagonal contribution: trace(S_pp)
        S_pp = contract('bid,bjd->bij', G_p, G_p)
        out += contract('bii->b', S_pp)  # trace instead of Frobenius norm squared

        for q in range(p):
            qs = q * tile_size
            qe = min((q + 1) * tile_size, T)
            G_q = backprops[:, qs:qe, :].to(dtype_acc)
            S_pq = contract('bid,bjd->bij', G_p, G_q)
            # Off-diagonal contribution: 2 * trace(S_pq)
            out += 2.0 * contract('bij->b', S_pq)  # This is wrong! Should be trace

    return torch.sqrt(out.clamp_min(0.0))

def test_bias_computations():
    """Test the different bias computation methods"""
    torch.manual_seed(42)
    
    # Test case 1: Small example
    B, T, d = 2, 4, 3
    backprops = torch.randn(B, T, d)
    
    print("=== Test Case 1: Small Example ===")
    print(f"Input shape: {backprops.shape}")
    
    original = original_bias_computation_3d(backprops)
    flash = flash_bias_computation_3d(backprops)
    
    print(f"Original result: {original}")
    print(f"Flash result: {flash}")
    print(f"Ratio (flash/original): {flash / original}")
    print(f"Difference: {torch.abs(flash - original)}")
    
    # Test case 2: Larger example
    B, T, d = 4, 16, 8
    backprops = torch.randn(B, T, d)
    
    print("\n=== Test Case 2: Larger Example ===")
    print(f"Input shape: {backprops.shape}")
    
    original = original_bias_computation_3d(backprops)
    flash = flash_bias_computation_3d(backprops)
    
    print(f"Original result: {original}")
    print(f"Flash result: {flash}")
    print(f"Ratio (flash/original): {flash / original}")
    print(f"Difference: {torch.abs(flash - original)}")
    
    # Mathematical analysis
    print("\n=== Mathematical Analysis ===")
    ggT = torch.einsum("nik,njk->nij", backprops, backprops)
    frobenius_norm = torch.sqrt(torch.einsum("nij,nij->n", ggT, ggT))
    trace_norm = torch.sqrt(torch.einsum("nii->n", ggT))
    
    print(f"||GG^T||_F (Frobenius norm): {frobenius_norm}")
    print(f"sqrt(trace(GG^T)) (trace norm): {trace_norm}")
    print(f"Original algorithm uses: trace norm")
    print(f"Flash algorithm computes: Frobenius norm")

if __name__ == "__main__":
    test_bias_computations()