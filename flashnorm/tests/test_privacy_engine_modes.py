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

import pytest
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from flashnorm.privacy_engine import FlashNormPrivacyEngine


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.conv = nn.Conv1d(8, 8, kernel_size=1)
        self.ln = nn.LayerNorm(8)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        h = self.embed(x)  # (B, T, E)
        h = self.conv(h.permute(0, 2, 1)).permute(0, 2, 1)
        h = self.ln(h)
        h = h.mean(dim=1)
        return self.fc(h)


@pytest.mark.parametrize(
    "mode",
    [
        "ghost",
        "ghost_bk",
        "flash",
        "flash_bk",
        "flash_fuse",
        "flash_fuse_bk",
    ],
)
def test_make_private_modes_cpu(mode):
    torch.manual_seed(0)
    model = _ToyModel()

    inputs = torch.randint(0, 16, (8, 3))
    targets = torch.randint(0, 4, (8,))
    data = TensorDataset(inputs, targets)
    data_loader = DataLoader(data, batch_size=4, shuffle=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss(reduction="mean")
    loss_reduction = "mean"
    orig_reduction = criterion.reduction

    engine = FlashNormPrivacyEngine()
    model, optimizer, criterion_dp, data_loader = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        criterion=criterion,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        loss_reduction=loss_reduction,
        poisson_sampling=True,
        grad_sample_mode=mode,
    )

    # Ensure loss_reduction is consistent across components
    assert (
        loss_reduction
        == orig_reduction
        == criterion_dp.loss_reduction
        == model.loss_reduction
        == optimizer.loss_reduction
    ), "loss_reduction should be the same across GradSampleModule, Optimizer, Criterion, and loss_reduction"

    model.train()
    for x, y in data_loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion_dp(logits, y)
        loss.backward()
        optimizer.step()
        break  # one step is enough to validate the plumbing

    eps = engine.get_epsilon(delta=1e-5)
    assert eps > 0


def test_mode_consistency_norm_and_loss_cpu():
    torch.manual_seed(0)
    modes = [
        "ghost",
        "ghost_bk",
        "flash",
        "flash_bk",
        "flash_fuse",
        "flash_fuse_bk",
    ]

    # Fixed dataset for deterministic comparison
    inputs = torch.randint(0, 16, (6, 3))
    targets = torch.randint(0, 4, (6,))
    data = TensorDataset(inputs, targets)

    reference_loss = None
    reference_norm = None

    # First run ghost to establish reference
    ref_mode = "ghost"
    torch.manual_seed(1234)
    model = _ToyModel()
    data_loader = DataLoader(data, batch_size=3, shuffle=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss(reduction="mean")
    engine = FlashNormPrivacyEngine()
    model, optimizer, criterion_dp, data_loader = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        criterion=criterion,
        noise_multiplier=0.5,
        max_grad_norm=1.0,
        loss_reduction="mean",
        poisson_sampling=False,
        grad_sample_mode=ref_mode,
    )
    batch = next(iter(data_loader))
    x, y = batch
    optimizer.zero_grad()
    logits = model(x)
    loss = criterion_dp(logits, y)
    reference_loss = F.cross_entropy(logits, y, reduction="mean")
    loss.backward()
    reference_norm = model.get_norm_sample().detach().cpu()

    # Compare other modes against ghost reference
    for mode in modes:
        torch.manual_seed(1234)  # reset model init for fair comparison
        model = _ToyModel()
        data_loader = DataLoader(data, batch_size=3, shuffle=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss(reduction="mean")

        engine = FlashNormPrivacyEngine()
        model, optimizer, criterion_dp, data_loader = engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            criterion=criterion,
            noise_multiplier=0.5,
            max_grad_norm=1.0,
            loss_reduction="mean",
            poisson_sampling=False,
            grad_sample_mode=mode,
        )

        batch = next(iter(data_loader))
        x, y = batch

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion_dp(logits, y)
        test_loss_value = F.cross_entropy(logits, y, reduction="mean")

        loss.backward()

        norm = model.get_norm_sample().detach().cpu()

        assert torch.allclose(test_loss_value, reference_loss, atol=1e-6, rtol=1e-6)
        assert torch.allclose(norm, reference_norm, atol=1e-5, rtol=1e-5)

