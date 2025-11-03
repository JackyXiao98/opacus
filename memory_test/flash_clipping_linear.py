from typing import Dict, List
import math
import torch
import torch.nn as nn
from opt_einsum import contract

@torch.no_grad()
def _flash_frobenius_inner_over_T(
    A: torch.Tensor,  # [B, T, d_a]
    G: torch.Tensor,  # [B, T, d_g]
    tile_size: int = 256,
    dtype_acc = torch.float32,
) -> torch.Tensor:
    B, T, d_a = A.shape
    _, _, d_g = G.shape
    ga = torch.zeros(B, dtype=dtype_acc, device=A.device)

    num_tiles = (T + tile_size - 1) // tile_size

    for p in range(num_tiles):
        ps = p * tile_size
        pe = min((p + 1) * tile_size, T)
        A_p = A[:, ps:pe, :].to(dtype_acc)
        G_p = G[:, ps:pe, :].to(dtype_acc)

        # diagonal block (p, p)
        Sg_pp = contract('bid,bjd->bij', G_p, G_p)  # [B, tau_p, tau_p]
        Sa_pp = contract('bid,bjd->bij', A_p, A_p)
        ga += contract('bij,bij->b', Sg_pp, Sa_pp)

        # off-diagonal blocks (q < p)
        for q in range(p):
            qs = q * tile_size
            qe = min((q + 1) * tile_size, T)
            A_q = A[:, qs:qe, :].to(dtype_acc)
            G_q = G[:, qs:qe, :].to(dtype_acc)

            Sg_pq = contract('bid,bjd->bij', G_p, G_q)  # [B, tau_p, tau_q]
            Sa_pq = contract('bid,bjd->bij', A_p, A_q)
            ga += 2.0 * contract('bij,bij->b', Sg_pq, Sa_pq)

    return ga

@torch.no_grad()
def _flash_sum_over_time_norm_squared(
    G: torch.Tensor,  # [B, T, d_g]
    tile_size: int = 256,
    dtype_acc = torch.float32,
) -> torch.Tensor:
    """
    Compute ||sum_t G[b,t,:]||_2^2 for each batch element.
    This matches the original algorithm: sqrt(sum_k (sum_t G[b,t,k])^2)
    
    The original algorithm computes: torch.einsum("n...i->n", ggT) where ggT = G @ G^T
    This is equivalent to: sum_k (sum_t G[b,t,k])^2
    """
    B, T, d_g = G.shape
    
    # Simple and efficient: just sum over time dimension, then compute L2 norm squared
    sum_over_time = torch.sum(G.to(dtype_acc), dim=1)  # [B, d_g]
    return torch.sum(sum_over_time * sum_over_time, dim=1)  # [B]

@torch.no_grad()
def compute_linear_norm_sample(
    layer: nn.Linear,
    activations: List[torch.Tensor],
    backprops: torch.Tensor,
    tile_size: int = 256,
    dtype_acc = torch.float32,
) -> Dict[nn.Parameter, torch.Tensor]:
    A = activations[0]
    ret: Dict[nn.Parameter, torch.Tensor] = {}

    if backprops.dim() == 2:
        if layer.weight.requires_grad:
            g2 = contract('bi,bi->b', backprops, backprops)
            a2 = contract('bi,bi->b', A, A)
            ret[layer.weight] = torch.sqrt((g2 * a2).clamp_min(0.0))

        if (layer.bias is not None) and layer.bias.requires_grad:
            g2 = contract('bi,bi->b', backprops, backprops)
            ret[layer.bias] = torch.sqrt(g2.clamp_min(0.0))

    elif backprops.dim() == 3:
        B, T, d_out = backprops.shape
        _, T_a, d_in = A.shape
        assert T == T_a, f"Mismatched sequence lengths: backprops T={T} vs activations T={T_a}"

        if layer.weight.requires_grad:
            ga = _flash_frobenius_inner_over_T(A, backprops, tile_size=tile_size, dtype_acc=dtype_acc)
            ret[layer.weight] = torch.sqrt(ga.clamp_min(0.0))

        if (layer.bias is not None) and layer.bias.requires_grad:
            gg = _flash_sum_over_time_norm_squared(backprops, tile_size=tile_size, dtype_acc=dtype_acc)
            ret[layer.bias] = torch.sqrt(gg.clamp_min(0.0))

    else:
        raise ValueError(f"Unsupported backprops dim: {backprops.dim()}")

    return ret
