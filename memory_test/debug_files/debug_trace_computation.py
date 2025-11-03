import torch
from opt_einsum import contract

def debug_trace_computation():
    """Debug the trace computation step by step"""
    torch.manual_seed(42)
    
    # Simple test case
    B, T, d = 2, 4, 3
    G = torch.randn(B, T, d)
    
    print("=== Debug Trace Computation ===")
    print(f"G shape: {G.shape}")
    
    # Method 1: Original algorithm
    ggT = torch.einsum("nik,njk->nij", G, G)  # [B, T, T]
    trace_original = torch.einsum("nii->n", ggT)  # trace for each batch
    print(f"Original trace(GG^T): {trace_original}")
    
    # Method 2: Direct computation
    trace_direct = torch.sum(G * G, dim=2).sum(dim=1)  # sum over features, then time
    print(f"Direct trace(GG^T): {trace_direct}")
    
    # Method 3: Tiled computation (current implementation)
    tile_size = 2
    B, T, d_g = G.shape
    out = torch.zeros(B, dtype=torch.float32, device=G.device)
    num_tiles = (T + tile_size - 1) // tile_size
    
    print(f"Number of tiles: {num_tiles}")
    
    for p in range(num_tiles):
        ps = p * tile_size
        pe = min((p + 1) * tile_size, T)
        G_p = G[:, ps:pe, :].to(torch.float32)
        print(f"Tile {p}: G_p shape {G_p.shape}, range [{ps}:{pe}]")
        
        # Diagonal block contribution
        S_pp = contract('bid,bjd->bij', G_p, G_p)
        trace_pp = contract('bii->b', S_pp)
        print(f"  Diagonal trace: {trace_pp}")
        out += trace_pp
        
        for q in range(p):
            qs = q * tile_size
            qe = min((q + 1) * tile_size, T)
            G_q = G[:, qs:qe, :].to(torch.float32)
            print(f"  Off-diagonal tile {q}: G_q shape {G_q.shape}, range [{qs}:{qe}]")
            
            # Off-diagonal block contribution
            S_pq = contract('bid,bjd->bij', G_p, G_q)
            print(f"    S_pq shape: {S_pq.shape}")
            
            # PROBLEM: We're computing trace of a non-square matrix!
            # S_pq is [B, tau_p, tau_q] where tau_p and tau_q might be different
            # trace only makes sense for square matrices
            
            # What we actually want is the sum of all elements in the cross-correlation
            trace_pq = contract('bij->b', S_pq)  # sum all elements
            print(f"    Cross-correlation sum: {trace_pq}")
            out += 2.0 * trace_pq
    
    print(f"Tiled trace result: {out}")
    
    # Method 4: Correct tiled computation
    print("\n=== Correct Tiled Computation ===")
    out_correct = torch.zeros(B, dtype=torch.float32, device=G.device)
    
    for p in range(num_tiles):
        ps = p * tile_size
        pe = min((p + 1) * tile_size, T)
        G_p = G[:, ps:pe, :]
        
        for q in range(num_tiles):
            qs = q * tile_size
            qe = min((q + 1) * tile_size, T)
            G_q = G[:, qs:qe, :]
            
            # Compute G_p @ G_q^T and sum diagonal elements
            cross_corr = contract('bid,bjd->bij', G_p, G_q)  # [B, tau_p, tau_q]
            
            if p == q:
                # Diagonal block: take trace
                diagonal_sum = contract('bii->b', cross_corr)
                out_correct += diagonal_sum
                print(f"Block ({p},{q}): diagonal trace = {diagonal_sum}")
            else:
                # Off-diagonal block: this contributes 0 to the trace of the full matrix
                # because trace(AB) ≠ sum of traces of blocks unless blocks are on diagonal
                print(f"Block ({p},{q}): off-diagonal, contributes 0 to trace")
    
    print(f"Correct tiled trace: {out_correct}")
    
    # The issue is conceptual: trace(GG^T) = sum of diagonal elements of the full matrix
    # When we tile, only the diagonal tiles contribute to the trace!

if __name__ == "__main__":
    debug_trace_computation()