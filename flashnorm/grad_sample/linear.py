#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Dict, List

import torch
import torch.nn as nn

from .utils import register_grad_sampler, register_norm_sampler
from .triton_kernels import compute_linear_norm_sample_flash, is_triton_available


logger = logging.getLogger(__name__)
logging.disabled = False


@register_grad_sampler(nn.Linear)
def compute_linear_grad_sample(
    layer: nn.Linear, activations: List[torch.Tensor], backprops: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for ``nn.Linear`` layer

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    activations = activations[0]

    activations = activations.to(backprops.dtype)

    ret = {}
    if layer.weight.requires_grad:
        gs = torch.einsum("n...i,n...j->nij", backprops, activations)
        ret[layer.weight] = gs
    if layer.bias is not None and layer.bias.requires_grad:
        ret[layer.bias] = torch.einsum("n...k->nk", backprops)
    return ret


@register_norm_sampler(nn.Linear)
def compute_linear_norm_sample(
    layer: nn.Linear, activations: List[torch.Tensor], backprops: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradient norms for ``nn.Linear`` layer

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    activations = activations[0]
    activations = activations.to(backprops.dtype)
    # print(layer, "\n", "activation shape: ", activations.shape, "backprop shape: ", backprops.shape)
    ret = {}

    if backprops.dim() == 2:
        if layer.weight.requires_grad:
            g = torch.einsum("n...i,n...i->n", backprops, backprops)
            a = torch.einsum("n...j,n...j->n", activations, activations)
            ret[layer.weight] = torch.sqrt((g * a).flatten())
        if layer.bias is not None and layer.bias.requires_grad:
            ret[layer.bias] = torch.sqrt(
                torch.einsum("n...i,n...i->n", backprops, backprops).flatten()
            )
    elif backprops.dim() == 3:
        if layer.weight.requires_grad:
            # Optimized computation: ||A^T @ G||_F^2 = <G@G^T, A@A^T>_F
            # This avoids creating large [n, T, T] intermediate tensors
            # A: activations [n, T, d_in], G: backprops [n, T, d_out]
            A_t = activations.transpose(1, 2)  # [n, d_in, T]
            S = torch.bmm(A_t, backprops)  # [n, d_in, d_out]
            ga = torch.sum(S * S, dim=(1, 2)).clamp(min=0)  # [n]
            ret[layer.weight] = torch.sqrt(ga)
        if layer.bias is not None and layer.bias.requires_grad:
            # For bias: ||G||_F^2 = sum(G * G) over all dimensions except batch
            ret[layer.bias] = torch.sqrt(
                torch.sum(backprops * backprops, dim=(1, 2)).clamp(min=0)
            )
    return ret


@register_norm_sampler(nn.Linear, "flash")
def compute_linear_norm_sample_flash_wrapper(
    layer: nn.Linear, activations: List[torch.Tensor], backprops: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Flash Clipping accelerated version of per sample gradient norms for ``nn.Linear`` layer.
    This function provides significant speedup for sequence models (3D tensors) by using
    optimized Triton kernels for gradient norm computations.

    Args:
        layer: Linear layer
        activations: Activations from forward pass
        backprops: Backpropagated gradients

    Returns:
        Dictionary mapping parameters to their gradient norms
    """

    if not is_triton_available():
        logger.warning(
            "Triton is not available. Falling back to standard norm computation. "
            "Install triton for better performance: pip install triton"
        )
        return compute_linear_norm_sample(layer, activations, backprops)
    
    return compute_linear_norm_sample_flash(layer, activations, backprops)