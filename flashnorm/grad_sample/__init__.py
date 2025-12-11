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

from flashnorm.grad_sample.conv import compute_conv_grad_sample, compute_conv_norm_sample, compute_conv_norm_sample_flash_wrapper  # noqa
from flashnorm.grad_sample.opacus_wrappers import (  # noqa
    compute_embedding_grad_sample,
    compute_embedding_norm_sample,
    compute_instance_norm_grad_sample,
    compute_layer_norm_grad_sample,
    compute_group_norm_grad_sample,
    compute_rnn_linear_grad_sample,
    compute_sequence_bias_grad_sample,
)
from flashnorm.grad_sample.embedding_norm_sample import compute_embedding_norm_sample as compute_embedding_norm_sample_flashnorm  # noqa
from flashnorm.grad_sample.grad_sample_module import GradSampleModule, create_or_accumulate_grad_sample
from flashnorm.grad_sample.grad_sample_module_fast_gradient_clipping import (  # noqa
    GradSampleModuleFastGradientClipping,
)
from flashnorm.grad_sample.grad_sample_module_fast_gradient_clipping_fsdp import (  # noqa
    GradSampleModuleFastGradientClippingFSDP,
)
from flashnorm.grad_sample.grad_sample_module_fast_gradient_clipping_fsdp_fuse import (  # noqa
    GradSampleModuleFastGradientClippingFSDPFuse,
)
from flashnorm.grad_sample.grad_sample_module_fast_gradient_clipping_fuse import (  # noqa
    GradSampleModuleFastGradientClippingFuse,
)
from flashnorm.grad_sample.fused_flash_linear import (  # noqa
    TRITON_AVAILABLE,
    FusedFlashLinear,
    replace_linear_with_fused,
    get_fused_linear_modules,
)
from flashnorm.grad_sample.triton_fused_kernel import (  # noqa
    TRITON_AVAILABLE as TRITON_KERNEL_AVAILABLE,
)
from opacus.grad_sample.grad_sample_module_fast_gradient_clipping_tp import (  # noqa
    GradSampleModuleFastGradientClippingTP,
)
from opacus.grad_sample.gsm_base import AbstractGradSampleModule
from opacus.grad_sample.gsm_exp_weights import GradSampleModuleExpandedWeights
from opacus.grad_sample.gsm_no_op import GradSampleModuleNoOp
from flashnorm.grad_sample.linear import compute_linear_grad_sample  # noqa
from flashnorm.grad_sample.rms_norm import compute_rms_norm_grad_sample  # noqa
from flashnorm.grad_sample.utils import (
    get_gsm_class,
    register_grad_sampler,
    register_norm_sampler,
    wrap_model,
)


__all__ = [
    "GradSampleModule",
    "GradSampleModuleFastGradientClipping",
    "GradSampleModuleFastGradientClippingFSDP",
    "GradSampleModuleFastGradientClippingFSDPFuse",
    "GradSampleModuleFastGradientClippingFuse",
    "GradSampleModuleFastGradientClippingTP",
    "GradSampleModuleExpandedWeights",
    "GradSampleModuleNoOp",
    "AbstractGradSampleModule",
    "register_grad_sampler",
    "register_norm_sampler",
    "create_or_accumulate_grad_sample",
    "wrap_model",
    "get_gsm_class",
    # opacus-registered samplers re-exposed for flashnorm registries
    "compute_embedding_grad_sample",
    "compute_embedding_norm_sample",
    "compute_instance_norm_grad_sample",
    "compute_layer_norm_grad_sample",
    "compute_group_norm_grad_sample",
    "compute_rnn_linear_grad_sample",
    "compute_sequence_bias_grad_sample",
    # Fused Flash Linear exports
    "TRITON_AVAILABLE",
    "FusedFlashLinear",
    "replace_linear_with_fused",
    "get_fused_linear_modules",
]
