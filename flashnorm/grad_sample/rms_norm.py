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


from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.utils.tensor_utils import sum_over_all_but_batch_and_last_n

from .utils import register_grad_sampler


@register_grad_sampler(nn.RMSNorm)
def compute_rms_norm_grad_sample(
    layer: nn.RMSNorm,
    activations: List[torch.Tensor],
    backprops: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for RMSNorm

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    activations = activations[0]
    ret = {}
    if layer.weight.requires_grad:
        ret[layer.weight] = sum_over_all_but_batch_and_last_n(
            F.rms_norm(activations, layer.normalized_shape, eps=layer.eps) * backprops,
            layer.weight.dim(),
        )
    return ret


# Support for HuggingFace transformers' LlamaRMSNorm and other custom RMSNorm implementations
def compute_custom_rms_norm_grad_sample(
    layer: nn.Module,
    activations: List[torch.Tensor],
    backprops: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for custom RMSNorm implementations (e.g., LlamaRMSNorm)
    
    This works for any RMSNorm-like layer that has:
    - A 'weight' parameter
    - Optional 'variance_epsilon' or 'eps' attribute
    
    Args:
        layer: RMSNorm layer (can be LlamaRMSNorm or similar)
        activations: Activations from forward pass
        backprops: Backpropagated gradients
    """
    activations = activations[0]
    ret = {}
    
    # Get weight parameter (might be named 'weight', 'scale', or 'gain')
    weight_param = getattr(layer, 'weight', None)
    if weight_param is not None and weight_param.requires_grad:
        # Get epsilon value (different names in different implementations)
        eps = getattr(layer, 'variance_epsilon', getattr(layer, 'eps', 1e-6))
        
        # Compute RMS normalization
        # RMS = sqrt(mean(x^2) + eps)
        # normalized = x / RMS
        variance = activations.pow(2).mean(-1, keepdim=True)
        normalized_activations = activations * torch.rsqrt(variance + eps)
        
        # Gradient for weight is normalized_input * backprop, summed over all but batch and param dims
        ret[weight_param] = sum_over_all_but_batch_and_last_n(
            normalized_activations * backprops,
            weight_param.dim(),
        )
    
    return ret


# Try to register HuggingFace's LlamaRMSNorm if available
try:
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    register_grad_sampler(LlamaRMSNorm)(compute_custom_rms_norm_grad_sample)
except ImportError:
    pass  # transformers not installed or LlamaRMSNorm not available
