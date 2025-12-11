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

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from flashnorm.grad_sample import (
    GradSampleModule,
    GradSampleModuleFastGradientClipping,
    compute_embedding_grad_sample,
    compute_embedding_norm_sample,
    compute_group_norm_grad_sample,
    compute_instance_norm_grad_sample,
    compute_layer_norm_grad_sample,
    compute_rnn_linear_grad_sample,
    compute_sequence_bias_grad_sample,
)
from flashnorm.privacy_engine import FlashNormPrivacyEngine


def test_opacus_wrappers_registered_in_flashnorm():
    # Verify samplers from opacus are registered in flashnorm registries
    samplers = GradSampleModule.GRAD_SAMPLERS
    fast_gc_samplers = GradSampleModuleFastGradientClipping.GRAD_SAMPLERS
    norm_samplers = GradSampleModuleFastGradientClipping.NORM_SAMPLERS

    for cls in [
        nn.Embedding,
        nn.MultiheadAttention,
        nn.RNNBase,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.LayerNorm,
        nn.Linear,
        nn.Conv1d,
    ]:
        assert cls in samplers or cls in fast_gc_samplers

    for cls in [nn.Embedding, nn.Conv1d, nn.Linear]:
        assert cls in norm_samplers

    # Ensure callables are imported (sanity)
    for fn in [
        compute_embedding_grad_sample,
        compute_embedding_norm_sample,
        compute_group_norm_grad_sample,
        compute_instance_norm_grad_sample,
        compute_layer_norm_grad_sample,
        compute_rnn_linear_grad_sample,
        compute_sequence_bias_grad_sample,
    ]:
        assert callable(fn)


class ToyGhostModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.conv = nn.Conv1d(8, 8, kernel_size=1)
        self.ln = nn.LayerNorm(8)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        # x: (batch, seq)
        h = self.embed(x)
        h = self.conv(h.permute(0, 2, 1)).permute(0, 2, 1)
        h = self.ln(h)
        h = h.mean(dim=1)
        return self.fc(h)


def test_ghost_training_uses_wrappers_cpu():
    torch.manual_seed(0)
    model = ToyGhostModel()

    inputs = torch.randint(0, 16, (12, 3))
    targets = torch.randint(0, 4, (12,))
    data = TensorDataset(inputs, targets)
    data_loader = DataLoader(data, batch_size=4, shuffle=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss(reduction="mean")

    engine = FlashNormPrivacyEngine()
    model, optimizer, criterion_dp, data_loader = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        loss_reduction="mean",
        poisson_sampling=True,
        grad_sample_mode="ghost",
    )

    assert criterion_dp.loss_reduction == "mean"
    assert model.loss_reduction == optimizer.loss_reduction == "mean"

    model.train()
    for x, y in data_loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion_dp(logits, y)
        loss.backward()
        optimizer.step()

    eps = engine.get_epsilon(delta=1e-5)
    assert eps > 0

