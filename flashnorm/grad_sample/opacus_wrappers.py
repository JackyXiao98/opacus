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

import torch.nn as nn

from opacus.grad_sample.dp_multihead_attention import (
    compute_sequence_bias_grad_sample as _opacus_compute_sequence_bias_grad_sample,
)
from opacus.grad_sample.dp_rnn import (
    compute_rnn_linear_grad_sample as _opacus_compute_rnn_linear_grad_sample,
)
from opacus.grad_sample.embedding import (
    compute_embedding_grad_sample as _opacus_compute_embedding_grad_sample,
    compute_embedding_norm_sample as _opacus_compute_embedding_norm_sample,
)
from opacus.grad_sample.group_norm import (
    compute_group_norm_grad_sample as _opacus_compute_group_norm_grad_sample,
)
from opacus.grad_sample.instance_norm import (
    compute_instance_norm_grad_sample as _opacus_compute_instance_norm_grad_sample,
)
from opacus.grad_sample.layer_norm import (
    compute_layer_norm_grad_sample as _opacus_compute_layer_norm_grad_sample,
)

from flashnorm.grad_sample.utils import register_grad_sampler, register_norm_sampler


@register_grad_sampler(nn.MultiheadAttention)
def compute_sequence_bias_grad_sample(module, activations, backprops):
    return _opacus_compute_sequence_bias_grad_sample(module, activations, backprops)


@register_grad_sampler(nn.RNNBase)
def compute_rnn_linear_grad_sample(module, activations, backprops):
    return _opacus_compute_rnn_linear_grad_sample(module, activations, backprops)


@register_grad_sampler(nn.Embedding)
def compute_embedding_grad_sample(module, activations, backprops):
    return _opacus_compute_embedding_grad_sample(module, activations, backprops)


@register_norm_sampler(nn.Embedding)
def compute_embedding_norm_sample(module, activations, backprops):
    return _opacus_compute_embedding_norm_sample(module, activations, backprops)


@register_grad_sampler(nn.GroupNorm)
def compute_group_norm_grad_sample(module, activations, backprops):
    return _opacus_compute_group_norm_grad_sample(module, activations, backprops)


@register_grad_sampler(nn.InstanceNorm1d)
@register_grad_sampler(nn.InstanceNorm2d)
@register_grad_sampler(nn.InstanceNorm3d)
def compute_instance_norm_grad_sample(module, activations, backprops):
    return _opacus_compute_instance_norm_grad_sample(module, activations, backprops)


@register_grad_sampler(nn.LayerNorm)
def compute_layer_norm_grad_sample(module, activations, backprops):
    return _opacus_compute_layer_norm_grad_sample(module, activations, backprops)

