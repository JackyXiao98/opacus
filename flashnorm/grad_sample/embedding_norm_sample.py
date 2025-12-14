#!/usr/bin/env python3
# Copyright 2024
#
# Safe embedding norm sampler that tolerates non-contiguous input tensors by
# using reshape instead of view. Kept separate to avoid modifying upstream
# Opacus sources.

from typing import Dict, List

import torch
from torch import nn


def compute_embedding_norm_sample(
    layer: nn.Embedding,
    activations: List[torch.Tensor],
    backprops: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per-sample gradient norms for nn.Embedding.

    Uses reshape in place of view to support non-contiguous inputs.
    """
    device = activations[0].device
    input_ids = activations[0].to(device)
    grad_values = backprops.to(device)

    # Reshape input_ids preserving batch first; reshape handles non-contiguous tensors.
    input_ids = input_ids.reshape(input_ids.shape[0], -1)

    # Flatten backprop values while keeping embedding dimension last.
    grad_values = grad_values.reshape(-1, grad_values.size(-1))

    nrows = input_ids.size(0)
    ncols = input_ids.size(1)
    row_indices = (
        torch.repeat_interleave(torch.arange(nrows, device=device), ncols)
        .unsqueeze(-1)
        .to(device)
    )

    # Pair row indices with flattened input ids (reshape to avoid view on non-contiguous)
    flattened_indices = input_ids.reshape(-1, 1)
    paired_indices = torch.cat([row_indices, flattened_indices], dim=1).to(device)

    unique_paired_indices, new_index_positions = torch.unique(
        paired_indices, dim=0, return_inverse=True, sorted=True
    )

    num_unique = unique_paired_indices.size(0)
    summed_gradients = torch.zeros(
        num_unique, grad_values.size(-1), device=device, dtype=grad_values.dtype
    )
    summed_gradients = summed_gradients.index_add(
        0, new_index_positions.to(device), grad_values
    )
    sqr_gradient_sum = torch.sum(summed_gradients**2, dim=1)

    result = torch.zeros(nrows, device=device, dtype=grad_values.dtype)
    unique_batch_ids = unique_paired_indices[:, 0].to(device)
    result.scatter_add_(0, unique_batch_ids, sqr_gradient_sum)

    result_sqrt = torch.sqrt(result)
    return {layer.weight: result_sqrt}

