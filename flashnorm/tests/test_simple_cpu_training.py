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

from flashnorm.privacy_engine import FlashNormPrivacyEngine


def test_flashnorm_privacy_engine_cpu_train_step():
    torch.manual_seed(0)

    model = nn.Linear(4, 2)
    data = TensorDataset(torch.randn(8, 4), torch.randint(0, 2, (8,)))
    data_loader = DataLoader(data, batch_size=4, shuffle=False)
    criterion = nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    privacy_engine = FlashNormPrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        loss_reduction="mean",
        poisson_sampling=True,
    )

    # Loss reduction must be consistent across components
    assert (
        "mean"
        == criterion.reduction
        == model.loss_reduction
        == optimizer.loss_reduction
    ), "loss_reduction should be the same across GradSampleModule, Optimizer, Criterion, and loss_reduction"

    model.train()
    for inputs, targets in data_loader:
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

    # Basic sanity: privacy accountant should return finite epsilon
    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    assert epsilon > 0

