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

from opacus.optimizers.adaclipoptimizer import AdaClipDPOptimizer
from opacus.optimizers.ddp_perlayeroptimizer import SimpleDistributedPerLayerOptimizer
from flashnorm.optimizers.ddpoptimizer import DistributedDPOptimizer
from flashnorm.optimizers.ddpoptimizer_fast_gradient_clipping import (
    DistributedDPOptimizerFastGradientClipping,
)
from opacus.optimizers.fsdpoptimizer_fast_gradient_clipping import (
    FSDPOptimizerFastGradientClipping,
)
from flashnorm.optimizers.optimizer import DPOptimizer
from flashnorm.optimizers.optimizer_fast_gradient_clipping import (
    DPOptimizerFastGradientClipping,
)
from opacus.optimizers.perlayeroptimizer import DPPerLayerOptimizer


__all__ = [
    "AdaClipDPOptimizer",
    "DistributedDPOptimizer",
    "DPOptimizer",
    "DPOptimizerFastGradientClipping",
    "DistributedDPOptimizerFastGradientlipping",
    "FSDPOptimizerFastGradientClipping",
    "DPPerLayerOptimizer",
    "SimpleDistributedPerLayerOptimizer",
]


def get_optimizer_class(clipping: str, distributed: bool, grad_sample_mode: str = None):
    if grad_sample_mode in ["ghost", "flash", "flash_bk", "ghost_bk", "flash_fuse", "flash_fuse_bk"]:
        if clipping == "flat" and distributed is False:
            return DPOptimizerFastGradientClipping
        elif clipping == "flat" and distributed is True:
            return DistributedDPOptimizerFastGradientClipping
        else:
            raise ValueError(
                f"Unsupported combination of parameters. Clipping: {clipping} and grad_sample_mode: {grad_sample_mode}"
            )
    elif grad_sample_mode in ["ghost_fsdp", "ghost_fsdp_bk", "flash_fsdp", "flash_fsdp_bk", "flash_fsdp_fuse", "flash_fsdp_fuse_bk"]:
        if clipping == "flat" and distributed is True:
            return FSDPOptimizerFastGradientClipping
        else:
            raise ValueError(
                f"Unsupported combination of parameters. Clipping: {clipping}, distributed: {distributed}, and grad_sample_mode: {grad_sample_mode}"
            )
    elif clipping == "flat" and distributed is False:
        return DPOptimizer
    elif clipping == "flat" and distributed is True:
        return DistributedDPOptimizer
    elif clipping == "per_layer" and distributed is False:
        return DPPerLayerOptimizer
    elif clipping == "per_layer" and distributed is True:
        if grad_sample_mode == "hooks" or grad_sample_mode == "ew":
            return SimpleDistributedPerLayerOptimizer
        else:
            raise ValueError(f"Unexpected grad_sample_mode: {grad_sample_mode}")
    elif clipping == "adaptive" and distributed is False:
        return AdaClipDPOptimizer
    raise ValueError(
        f"Unexpected optimizer parameters. Clipping: {clipping}, distributed: {distributed}"
    )
