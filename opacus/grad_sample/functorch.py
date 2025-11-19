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

import copy
from contextlib import nullcontext

import torch
import torch.nn as nn
from opacus.layers.dp_rnn import RNNLinear
from torch.func import grad, vmap


# https://gist.github.com/zou3519/7769506acc899d83ef1464e28f22e6cf
def make_functional(mod: nn.Module, disable_autograd_tracking: bool = False):
    """
    Helper method to mimic deprecated `functorch.make_functional()` behaviour. See
    https://pytorch.org/docs/master/func.migrating.html

    Args:
        mod: module to be converted to functional
        disable_autograd_tracking:

    Returns:
        Tuple with cloned model and new params
    """
    params_dict = dict(mod.named_parameters())
    params_names = params_dict.keys()
    params_values = tuple(params_dict.values())

    stateless_mod = copy.deepcopy(mod)
    stateless_mod.to("meta")

    if hasattr(stateless_mod, "allow_grad_accumulation"):
        stateless_mod.allow_grad_accumulation()

    def fmodel(new_params_values, *args, **kwargs):
        new_params_dict = {
            name: value for name, value in zip(params_names, new_params_values)
        }
        return torch.func.functional_call(stateless_mod, new_params_dict, args, kwargs)

    if disable_autograd_tracking:
        params_values = torch.utils._pytree.tree_map(torch.Tensor.detach, params_values)
    return fmodel, params_values


def prepare_layer(layer, batch_first=True):
    """
    Prepare a layer to compute grad samples using functorch.
    The grad samples are computed by redoing the forward and
    backward passes on the functional version of the module.

    Args:
        layer: the layer to prepare
        batch_first: whether the input is batch_first or not
    """
    if len(list(layer.buffers())) > 0:
        raise NotImplementedError(
            "This layer has buffers and is not supported by Opacus"
        )
    if type(layer) is nn.EmbeddingBag:
        raise NotImplementedError("Functorch does not support EmbeddingBag yet")

    flayer, _ = make_functional(layer)

    def compute_loss_stateless_model(params, activations, backprops):
        # Handle both single tensor and tuple of tensors for activations
        if isinstance(activations, (tuple, list)):
            # Multiple inputs: expand batch dimension for each
            if batch_first or type(layer) is RNNLinear:
                batched_activations = tuple(act.unsqueeze(0) for act in activations)
                batched_backprops = backprops.unsqueeze(0)
            else:
                # If batch_first is False, the batch dimension is the second dimension
                batched_activations = tuple(act.unsqueeze(1) for act in activations)
                batched_backprops = backprops.unsqueeze(1)
            
            # mixed precision logic - check first activation
            first_activation = activations[0]
            is_mixed = first_activation.dtype != params[0].dtype
            mixed_lowest_dtype = first_activation.dtype
            device_type = first_activation.device.type
        else:
            # Single input: original behavior
            if batch_first or type(layer) is RNNLinear:
                batched_activations = activations.unsqueeze(0)
                batched_backprops = backprops.unsqueeze(0)
            else:
                # If batch_first is False, the batch dimension is the second dimension
                batched_activations = activations.unsqueeze(1)
                batched_backprops = backprops.unsqueeze(1)

            # mixed precision logic
            is_mixed = activations.dtype != params[0].dtype
            mixed_lowest_dtype = activations.dtype
            device_type = activations.device.type

        # use amp context if user is using mixed_precision, else proceed as usual
        with (
            torch.amp.autocast(device_type=device_type, dtype=mixed_lowest_dtype)
            if is_mixed
            else nullcontext()
        ):
            # Call flayer with unpacked activations if it's a tuple
            if isinstance(batched_activations, tuple):
                output = flayer(params, *batched_activations)
            else:
                output = flayer(params, batched_activations)
            
            # Handle tuple outputs (some models return tuples instead of tensors)
            if isinstance(output, tuple):
                # Take the first element if it's a tuple
                output = output[0]
            
            loss = (output * batched_backprops).sum()
        return loss

    ft_compute_grad = grad(compute_loss_stateless_model)
    # Note that the vmap is done on the first dimension, regardless of batch_first
    # This is because the activations and backprops given by the GradSampleModule
    # are always batch_first=True
    # For multi-input models, we need to vmap over each activation tensor
    # Use randomness='different' to support models with dropout/random operations
    layer.ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0, 0), randomness='different')


def ft_compute_per_sample_gradient(layer, activations, backprops):
    """
    Compute the per-sample gradient of the layer.
    Args:
        layer: the layer on which to compute the gradient
        activations: the input to the layer (can be a single tensor or tuple of tensors)
        backprops: the  gradient of the loss w.r.t. outputs of the layer
    """
    parameters = list(layer.parameters(recurse=True))
    if not hasattr(layer, "ft_compute_sample_grad"):
        prepare_layer(layer)

    # Handle both single activation and multiple activations
    if len(activations) == 1:
        # Single input case (backward compatible)
        activations_data = activations[0]
        if activations_data.dtype != backprops.dtype and activations_data.is_floating_point():
            activations_data = activations_data.to(backprops.dtype)
    else:
        # Multiple inputs case - convert types if needed
        activations_list = []
        for act in activations:
            if act.dtype != backprops.dtype and act.is_floating_point():
                activations_list.append(act.to(backprops.dtype))
            else:
                activations_list.append(act)
        activations_data = tuple(activations_list)
    
    per_sample_grads = layer.ft_compute_sample_grad(parameters, activations_data, backprops)

    ret = {}
    for i_p, p in enumerate(parameters):
        ret[p] = per_sample_grads[i_p]

    return ret
