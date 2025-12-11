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

import math
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.utils.tensor_utils import unfold2d, unfold3d

from .utils import register_grad_sampler, register_norm_sampler
from .triton_kernels import is_triton_available
import logging

logger = logging.getLogger(__name__)
logging.disabled = False


@register_grad_sampler([nn.Conv1d, nn.Conv2d, nn.Conv3d])
def compute_conv_grad_sample(
    layer: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d],
    activations: List[torch.Tensor],
    backprops: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for convolutional layers.

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    activations = activations[0]

    activations = activations.to(backprops.dtype)

    n = activations.shape[0]
    if n == 0:
        # Empty batch
        ret = {}
        ret[layer.weight] = torch.zeros_like(layer.weight).unsqueeze(0)
        if layer.bias is not None and layer.bias.requires_grad:
            ret[layer.bias] = torch.zeros_like(layer.bias).unsqueeze(0)
        return ret

    # FSDPWrapper adds a prefix 'FSDP' to layer type, e.g. FSDPConv2d.
    # Therefore the layer type can not be directly determined by type(layer).
    layer_type = (
        layer.__class__.__bases__[1]
        if isinstance(layer, torch.distributed.fsdp.FSDPModule)
        else type(layer)
    )
    # get activations and backprops in shape depending on the Conv layer
    if layer_type is nn.Conv2d:
        activations = unfold2d(
            activations,
            kernel_size=layer.kernel_size,
            padding=layer.padding,
            stride=layer.stride,
            dilation=layer.dilation,
        )
    elif layer_type is nn.Conv1d:
        activations = activations.unsqueeze(-2)  # add the H dimension
        # set arguments to tuples with appropriate second element
        if layer.padding == "same":
            total_pad = layer.dilation[0] * (layer.kernel_size[0] - 1)
            left_pad = math.floor(total_pad / 2)
            right_pad = total_pad - left_pad
        elif layer.padding == "valid":
            left_pad, right_pad = 0, 0
        else:
            left_pad, right_pad = layer.padding[0], layer.padding[0]
        activations = F.pad(activations, (left_pad, right_pad))
        activations = torch.nn.functional.unfold(
            activations,
            kernel_size=(1, layer.kernel_size[0]),
            stride=(1, layer.stride[0]),
            dilation=(1, layer.dilation[0]),
        )
    elif layer_type is nn.Conv3d:
        activations = unfold3d(
            activations,
            kernel_size=layer.kernel_size,
            padding=layer.padding,
            stride=layer.stride,
            dilation=layer.dilation,
        )
    backprops = backprops.reshape(n, -1, activations.shape[-1])

    ret = {}
    if layer.weight.requires_grad:
        # n=batch_sz; o=num_out_channels; p=(num_in_channels/groups)*kernel_sz
        grad_sample = torch.einsum("noq,npq->nop", backprops, activations)
        # rearrange the above tensor and extract diagonals.
        grad_sample = grad_sample.view(
            n,
            layer.groups,
            -1,
            layer.groups,
            int(layer.in_channels / layer.groups),
            np.prod(layer.kernel_size),
        )
        grad_sample = torch.einsum("ngrg...->ngr...", grad_sample).contiguous()
        shape = [n] + list(layer.weight.shape)
        ret[layer.weight] = grad_sample.view(shape)

    if layer.bias is not None and layer.bias.requires_grad:
        ret[layer.bias] = torch.sum(backprops, dim=2)

    return ret


# @register_grad_sampler([nn.Conv2d])
def convolution2d_backward_as_a_convolution(
    layer: nn.Conv2d,
    activations: List[torch.Tensor],
    backprops: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for Conv2d layers using backward.
    This is an alternative implementation and is not used because it is slower in many contexts.

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    activations = activations[0]
    batch_size = activations.shape[0]
    input_size = activations.shape[1]
    output_size = backprops.shape[1]

    # activations has shape (B, I, H, W)
    # backprops has shape (B, O, H, W)
    activations_ = activations.view(
        batch_size,
        layer.groups,
        input_size // layer.groups,
        activations.shape[2],
        activations.shape[3],
    )  # (B, G, I/G, H, W)

    activations_ = activations_.view(
        activations_.shape[0] * activations_.shape[1],
        activations_.shape[2],
        activations_.shape[3],
        activations_.shape[4],
    )  # (B*G, I / G, H, W)
    activations_ = activations_.transpose(0, 1)  # (I / G, B * G, H, W)
    backprops_ = backprops.view(
        backprops.shape[0] * backprops.shape[1],
        1,
        backprops.shape[2],
        backprops.shape[3],
    )  # (B*O, 1, H, W)

    # Without groups (I, B, H, W) X (B*O, 1, H, W) -> (I, B*O, H, W)
    # With groups (I / G, B*G, H, W) X (B*O, 1, H, W) -> (I / G, B * O, H, W)
    weight_grad_sample = F.conv2d(
        activations_,
        backprops_,
        bias=None,
        dilation=layer.stride,
        padding=layer.padding,
        stride=layer.dilation,
        groups=batch_size * layer.groups,
    )
    weight_grad_sample = weight_grad_sample.view(
        input_size // layer.groups,
        batch_size,
        output_size,
        *weight_grad_sample.shape[-2:]
    )  # (I / G, B, O, H, W)
    weight_grad_sample = weight_grad_sample.movedim(0, 2)  # (B, O, I/G, H, W)
    weight_grad_sample = weight_grad_sample[
        :, :, :, : layer.weight.shape[2], : layer.weight.shape[3]
    ]

    ret = {layer.weight: weight_grad_sample}
    if layer.bias is not None:
        ret[layer.bias] = torch.sum(backprops, dim=[-1, -2])

    return ret


@register_norm_sampler([nn.Conv1d, nn.Conv2d, nn.Conv3d])
def compute_conv_norm_sample(
    layer: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d],
    activations: List[torch.Tensor],
    backprops: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradient norms for convolutional layers.
    
    This function efficiently computes the norm of per-sample gradients without
    materializing the full gradient tensors, following the formula:
    ||grad|| = sqrt(||backprops||^2 * ||activations||^2)
    
    Args:
        layer: Convolutional layer (Conv1d, Conv2d, or Conv3d)
        activations: Activations from forward pass
        backprops: Backpropagated gradients
        
    Returns:
        Dictionary mapping parameters to their gradient norms
    """
    activations = activations[0]
    activations = activations.to(backprops.dtype)
    
    n = activations.shape[0]
    if n == 0:
        # Empty batch
        ret = {}
        if layer.weight.requires_grad:
            ret[layer.weight] = torch.zeros(0, device=backprops.device, dtype=backprops.dtype)
        if layer.bias is not None and layer.bias.requires_grad:
            ret[layer.bias] = torch.zeros(0, device=backprops.device, dtype=backprops.dtype)
        return ret
    
    # Determine layer type (handle FSDP wrapper case)
    layer_type = (
        layer.__class__.__bases__[1]
        if isinstance(layer, torch.distributed.fsdp.FSDPModule)
        else type(layer)
    )
    
    # Unfold activations depending on the Conv layer type
    # This transforms spatial convolution into matrix multiplication
    if layer_type is nn.Conv2d:
        activations = unfold2d(
            activations,
            kernel_size=layer.kernel_size,
            padding=layer.padding,
            stride=layer.stride,
            dilation=layer.dilation,
        )
    elif layer_type is nn.Conv1d:
        activations = activations.unsqueeze(-2)  # add the H dimension
        # set arguments to tuples with appropriate second element
        if layer.padding == "same":
            total_pad = layer.dilation[0] * (layer.kernel_size[0] - 1)
            left_pad = math.floor(total_pad / 2)
            right_pad = total_pad - left_pad
        elif layer.padding == "valid":
            left_pad, right_pad = 0, 0
        else:
            left_pad, right_pad = layer.padding[0], layer.padding[0]
        activations = F.pad(activations, (left_pad, right_pad))
        activations = torch.nn.functional.unfold(
            activations,
            kernel_size=(1, layer.kernel_size[0]),
            stride=(1, layer.stride[0]),
            dilation=(1, layer.dilation[0]),
        )
    elif layer_type is nn.Conv3d:
        activations = unfold3d(
            activations,
            kernel_size=layer.kernel_size,
            padding=layer.padding,
            stride=layer.stride,
            dilation=layer.dilation,
        )
    
    # Reshape backprops: (batch, out_channels, *spatial) -> (batch, out_channels, spatial_flat)
    backprops = backprops.reshape(n, -1, activations.shape[-1])
    
    ret = {}
    
    if layer.weight.requires_grad:
        # activations: (n, p, q) where p=(in_channels/groups)*kernel_size, q=output_spatial
        # backprops: (n, o, q) where o=out_channels
        
        # For grouped convolutions, compute norm per group and sum
        if layer.groups > 1:
            out_channels_per_group = layer.out_channels // layer.groups
            in_channels_per_group = layer.in_channels // layer.groups
            kernel_size_flat = int(np.prod(layer.kernel_size))
            p_per_group = in_channels_per_group * kernel_size_flat
            
            norm_sqr = torch.zeros(n, device=backprops.device, dtype=backprops.dtype)
            
            for g in range(layer.groups):
                # Extract backprops and activations for this group
                o_start = g * out_channels_per_group
                o_end = (g + 1) * out_channels_per_group
                p_start = g * p_per_group
                p_end = (g + 1) * p_per_group
                
                backprops_g = backprops[:, o_start:o_end, :]  # (n, out_per_group, q)
                activations_g = activations[:, p_start:p_end, :]  # (n, p_per_group, q)
                
                # Compute ggT and aaT for this group
                ggT_g = torch.einsum("noq,nom->nqm", backprops_g, backprops_g)
                aaT_g = torch.einsum("npq,npm->nqm", activations_g, activations_g)
                
                # Add this group's contribution to the total norm squared
                norm_sqr += torch.einsum("nqm,nqm->n", ggT_g, aaT_g)
            
            ret[layer.weight] = torch.sqrt(norm_sqr.clamp(min=0))
        else:
            # Non-grouped case (groups=1)
            # The norm squared computation for conv follows:
            # ||grad||^2 = sum_{o,c,k} (sum_q backprop[o,q] * activation[c*K+k,q])^2
            #            = tr(ggT @ aaT)
            # where:
            #   ggT[q1,q2] = sum_o backprop[o,q1] * backprop[o,q2]  (spatial correlation from backprops)
            #   aaT[q1,q2] = sum_{c,k} activation[c*K+k,q1] * activation[c*K+k,q2]  (spatial correlation from activations)
            
            # Compute ggT: sum over output channels -> (n, q, q)
            ggT = torch.einsum("noq,nom->nqm", backprops, backprops)
            
            # Compute aaT: sum over input channels and kernel positions -> (n, q, q)
            aaT = torch.einsum("npq,npm->nqm", activations, activations)
            
            # Compute norm squared: tr(ggT @ aaT) = sum_{q1,q2} ggT[q1,q2] * aaT[q1,q2]
            norm_sqr = torch.einsum("nqm,nqm->n", ggT, aaT).clamp(min=0)
            
            ret[layer.weight] = torch.sqrt(norm_sqr)
    
    if layer.bias is not None and layer.bias.requires_grad:
        # Bias gradient: grad[o] = sum_q backprops[o, q]
        # Norm squared: ||grad||^2 = sum_o (sum_q backprops[o, q])^2
        # First sum over spatial dimension (q)
        bias_grad = backprops.sum(dim=2)  # (n, o)
        # Then compute norm over output channels
        bias_norm_sqr = torch.einsum("no,no->n", bias_grad, bias_grad)
        ret[layer.bias] = torch.sqrt(bias_norm_sqr.clamp(min=0))
    
    return ret


def compute_conv_norm_sample_flash(
    layer: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d],
    activations: List[torch.Tensor],
    backprops: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Flash Clipping algorithm for computing per sample gradient norms for convolutional layers.
    
    This function efficiently computes the norm of per-sample gradients without
    materializing large (spatial, spatial) correlation matrices. Instead of computing
    ggT and aaT matrices (both of size q x q where q is the spatial dimension), 
    it directly computes the gradient matrix and its Frobenius norm.
    
    Algorithm:
    - Ghost clipping: ||grad||^2 = tr(ggT @ aaT) where ggT, aaT are (q, q) matrices
    - Flash clipping: ||grad||^2 = ||M||_F^2 where M = backprops @ activations.T is (o, p) matrix
    
    For Conv2d with 32x32 output, ghost clipping creates 1024x1024 matrices,
    while flash clipping only creates (out_channels, in_channels*kernel_size) matrices.
    
    Args:
        layer: Convolutional layer (Conv1d, Conv2d, or Conv3d)
        activations: Activations from forward pass
        backprops: Backpropagated gradients
        
    Returns:
        Dictionary mapping parameters to their gradient norms
    """
    activations = activations[0]
    activations = activations.to(backprops.dtype)
    
    n = activations.shape[0]
    if n == 0:
        # Empty batch
        ret = {}
        if layer.weight.requires_grad:
            ret[layer.weight] = torch.zeros(0, device=backprops.device, dtype=backprops.dtype)
        if layer.bias is not None and layer.bias.requires_grad:
            ret[layer.bias] = torch.zeros(0, device=backprops.device, dtype=backprops.dtype)
        return ret
    
    # Determine layer type (handle FSDP wrapper case)
    layer_type = (
        layer.__class__.__bases__[1]
        if isinstance(layer, torch.distributed.fsdp.FSDPModule)
        else type(layer)
    )
    
    # Unfold activations depending on the Conv layer type
    # This transforms spatial convolution into matrix multiplication
    if layer_type is nn.Conv2d:
        activations = unfold2d(
            activations,
            kernel_size=layer.kernel_size,
            padding=layer.padding,
            stride=layer.stride,
            dilation=layer.dilation,
        )
    elif layer_type is nn.Conv1d:
        activations = activations.unsqueeze(-2)  # add the H dimension
        # set arguments to tuples with appropriate second element
        if layer.padding == "same":
            total_pad = layer.dilation[0] * (layer.kernel_size[0] - 1)
            left_pad = math.floor(total_pad / 2)
            right_pad = total_pad - left_pad
        elif layer.padding == "valid":
            left_pad, right_pad = 0, 0
        else:
            left_pad, right_pad = layer.padding[0], layer.padding[0]
        activations = F.pad(activations, (left_pad, right_pad))
        activations = torch.nn.functional.unfold(
            activations,
            kernel_size=(1, layer.kernel_size[0]),
            stride=(1, layer.stride[0]),
            dilation=(1, layer.dilation[0]),
        )
    elif layer_type is nn.Conv3d:
        activations = unfold3d(
            activations,
            kernel_size=layer.kernel_size,
            padding=layer.padding,
            stride=layer.stride,
            dilation=layer.dilation,
        )
    
    # Reshape backprops: (batch, out_channels, *spatial) -> (batch, out_channels, spatial_flat)
    backprops = backprops.reshape(n, -1, activations.shape[-1])
    
    ret = {}
    
    if layer.weight.requires_grad:
        # activations: (n, p, q) where p=(in_channels/groups)*kernel_size, q=output_spatial
        # backprops: (n, o, q) where o=out_channels
        
        # Flash Clipping Algorithm: Compute norm directly without materializing (q, q) matrices
        # The gradient is: grad[o,p] = sum_q backprop[o,q] * activation[p,q]
        # Norm squared: ||grad||^2 = sum_{o,p} (sum_q backprop[o,q] * activation[p,q])^2
        #                          = ||M||_F^2 where M[o,p] = sum_q backprop[o,q] * activation[p,q]
        
        # For grouped convolutions, compute norm per group and sum
        if layer.groups > 1:
            out_channels_per_group = layer.out_channels // layer.groups
            in_channels_per_group = layer.in_channels // layer.groups
            kernel_size_flat = int(np.prod(layer.kernel_size))
            p_per_group = in_channels_per_group * kernel_size_flat
            
            norm_sqr = torch.zeros(n, device=backprops.device, dtype=backprops.dtype)
            
            for g in range(layer.groups):
                # Extract backprops and activations for this group
                o_start = g * out_channels_per_group
                o_end = (g + 1) * out_channels_per_group
                p_start = g * p_per_group
                p_end = (g + 1) * p_per_group
                
                backprops_g = backprops[:, o_start:o_end, :]  # (n, out_per_group, q)
                activations_g = activations[:, p_start:p_end, :]  # (n, p_per_group, q)
                
                # Flash clipping: Compute M = sum_q backprops_g[o,q] * activations_g[p,q]
                # M shape: (n, out_per_group, p_per_group)
                M_g = torch.einsum("noq,npq->nop", backprops_g, activations_g)
                
                # Add this group's contribution: ||M_g||_F^2
                norm_sqr += torch.einsum("nop,nop->n", M_g, M_g)
            
            ret[layer.weight] = torch.sqrt(norm_sqr.clamp(min=0))
        else:
            # Non-grouped case (groups=1)
            # Flash clipping: directly compute the gradient matrix and its norm
            # M[o,p] = sum_q backprop[o,q] * activation[p,q]
            # Shape: (n, out_channels, in_channels*kernel_size)
            M = torch.einsum("noq,npq->nop", backprops, activations)
            
            # Compute Frobenius norm squared: ||M||_F^2 = sum_{o,p} M[o,p]^2
            norm_sqr = torch.einsum("nop,nop->n", M, M).clamp(min=0)
            
            ret[layer.weight] = torch.sqrt(norm_sqr)
    
    if layer.bias is not None and layer.bias.requires_grad:
        # Bias gradient: grad[o] = sum_q backprops[o, q]
        # Norm squared: ||grad||^2 = sum_o (sum_q backprops[o, q])^2
        # First sum over spatial dimension (q)
        bias_grad = backprops.sum(dim=2)  # (n, o)
        # Then compute norm over output channels
        bias_norm_sqr = torch.einsum("no,no->n", bias_grad, bias_grad)
        ret[layer.bias] = torch.sqrt(bias_norm_sqr.clamp(min=0))
    
    return ret


@register_norm_sampler([nn.Conv2d], "flash")
def compute_conv_norm_sample_flash_wrapper(
    layer: nn.Conv2d,
    activations: List[torch.Tensor],
    backprops: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Flash Clipping accelerated version of per sample gradient norms for Conv2d layer.
    
    Uses flash clipping algorithm that avoids materializing large (spatial, spatial)
    correlation matrices by directly computing the gradient matrix via einsum operations.
    
    Args:
        layer: Conv2d layer
        activations: Activations from forward pass
        backprops: Backpropagated gradients
        
    Returns:
        Dictionary mapping parameters to their gradient norms
    """
    if not is_triton_available():
        logger.debug(
            "Triton is not available. Using PyTorch flash clipping implementation. "
            "Install triton for potential future performance improvements: pip install triton"
        )
    
    return compute_conv_norm_sample_flash(layer, activations, backprops)
