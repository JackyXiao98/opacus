# !/usr/bin/env python3
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

from typing import Sequence, Type, Union

import torch.nn as nn

from .grad_sample_module import GradSampleModule
from .grad_sample_module_fast_gradient_clipping import (
    GradSampleModuleFastGradientClipping,
)
from .grad_sample_module_fast_gradient_clipping_fsdp import (
    GradSampleModuleFastGradientClippingFSDP,
)
from .grad_sample_module_fast_gradient_clipping_fsdp_async import (
    GradSampleModuleFastGradientClippingFSDPAsync,
)
from .gsm_base import AbstractGradSampleModule
from .gsm_exp_weights import GradSampleModuleExpandedWeights
from .gsm_no_op import GradSampleModuleNoOp


def register_grad_sampler(
    target_class_or_classes: Union[Type[nn.Module], Sequence[Type[nn.Module]]],
):
    """
    Registers the decorated function as the ``grad_sampler`` of ``target_class_or_classes``, which is
    the function that will be invoked every time you want to compute a per-sample gradient
    of ``target_class_or_classes``. The signature of every grad_sampler is always the same:

    >>> @register_grad_sampler(MyCustomModel)
    ... def compute_grad_sample(module, activations, backprops):
    ...    pass

    It may help you to take a look at the existing grad_samplers inside Opacus, under ``opacus.grad_sample.``
    """

    def decorator(f):
        target_classes = (
            target_class_or_classes
            if isinstance(target_class_or_classes, Sequence)
            else [target_class_or_classes]
        )
        for target_class in target_classes:
            GradSampleModule.GRAD_SAMPLERS[target_class] = f
            GradSampleModuleFastGradientClipping.GRAD_SAMPLERS[target_class] = f
        return f

    return decorator


def register_norm_sampler(
    target_class_or_classes: Union[Type[nn.Module], Sequence[Type[nn.Module]]],
    mode: str = "default",
):
    """
    Registers the decorated function as the ``norm_sampler`` of ``target_class_or_classes``, which is
    the function that will be invoked every time you want to compute a per-sample gradient norm
    of ``target_class_or_classes``. The signature of every norm_sampler is always the same:

    >>> @register_norm_sampler(MyCustomModel)
    ... def compute_grad_norm_sample(module, activations, backprops):
    ...    pass
    
    Args:
        target_class_or_classes: The target class(es) to register the sampler for
        mode: The mode for the sampler ("default" or "flash")
    """

    def decorator(f):
        target_classes = (
            target_class_or_classes
            if isinstance(target_class_or_classes, Sequence)
            else [target_class_or_classes]
        )
        for target_class in target_classes:
            if mode == "flash":
                # Store flash samplers separately
                if not hasattr(GradSampleModuleFastGradientClipping, "FLASH_NORM_SAMPLERS"):
                    GradSampleModuleFastGradientClipping.FLASH_NORM_SAMPLERS = {}
                GradSampleModuleFastGradientClipping.FLASH_NORM_SAMPLERS[target_class] = f
            else:
                GradSampleModuleFastGradientClipping.NORM_SAMPLERS[target_class] = f
        return f

    return decorator


def wrap_model(model: nn.Module, grad_sample_mode: str, *args, **kwargs):
    cls = get_gsm_class(grad_sample_mode)
    if grad_sample_mode == "functorch":
        kwargs["force_functorch"] = True
    elif grad_sample_mode == "flash":
        kwargs["use_flash_clipping"] = True
        kwargs["use_ghost_clipping"] = True
    elif grad_sample_mode == "flash_bk":
        kwargs["use_flash_clipping"] = True
        kwargs["use_ghost_clipping"] = True
        kwargs["enable_fastdp_bookkeeping"] = True
    elif grad_sample_mode == "flash_fsdp":
        kwargs["use_flash_clipping"] = True
        kwargs["use_ghost_clipping"] = True
    elif grad_sample_mode == "ghost_bk":
        kwargs["use_ghost_clipping"] = True
        kwargs["enable_fastdp_bookkeeping"] = True
    elif grad_sample_mode == "ghost_fsdp_bk":
        kwargs["use_ghost_clipping"] = True
        kwargs["enable_fastdp_bookkeeping"] = True
    elif grad_sample_mode == "flash_fsdp_bk":
        kwargs["use_flash_clipping"] = True
        kwargs["use_ghost_clipping"] = True
        kwargs["enable_fastdp_bookkeeping"] = True
    return cls(model, *args, **kwargs)


def get_gsm_class(grad_sample_mode: str) -> Type[AbstractGradSampleModule]:
    """
    Returns AbstractGradSampleModule subclass correspinding to the input mode.
    See README for detailed comparison between grad sample modes.

    :param grad_sample_mode:
    :return:
    """
    if grad_sample_mode in ["hooks", "functorch"]:
        return GradSampleModule
    elif grad_sample_mode == "ew":
        return GradSampleModuleExpandedWeights
    elif grad_sample_mode == "ghost":
        return GradSampleModuleFastGradientClipping
    elif grad_sample_mode == "flash":
        return GradSampleModuleFastGradientClipping
    elif grad_sample_mode == "flash_bk":
        return GradSampleModuleFastGradientClipping
    elif grad_sample_mode == "ghost_bk":
        return GradSampleModuleFastGradientClipping
    elif grad_sample_mode == "ghost_fsdp":
        return GradSampleModuleFastGradientClippingFSDP
    elif grad_sample_mode == "ghost_fsdp_bk":
        return GradSampleModuleFastGradientClippingFSDP
    elif grad_sample_mode == "flash_fsdp":
        return GradSampleModuleFastGradientClippingFSDPAsync
    elif grad_sample_mode == "flash_fsdp_bk":
        return GradSampleModuleFastGradientClippingFSDPAsync
    elif grad_sample_mode == "no_op":
        return GradSampleModuleNoOp
    else:
        raise ValueError(
            f"Unexpected grad_sample_mode: {grad_sample_mode}. "
            f"Allowed values: hooks, ew, ghost, flash, flash_bk, ghost_bk, ghost_fsdp, ghost_fsdp_bk, flash_fsdp, flash_fsdp_bk, no_op"
        )
