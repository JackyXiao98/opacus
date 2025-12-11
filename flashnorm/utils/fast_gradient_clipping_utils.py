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

from flashnorm.grad_sample.grad_sample_module_fast_gradient_clipping import (
    GradSampleModuleFastGradientClipping,
)
from flashnorm.optimizers import DPOptimizerFastGradientClipping


def _is_fsdp_model(module) -> bool:
    """
    Check if model is wrapped with FSDP.
    
    Returns True if the module or any of its submodules is wrapped with FSDP,
    or if it's an instance of GradSampleModuleFastGradientClippingFSDP or
    GradSampleModuleFastGradientClippingFSDPFuse.
    """
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
        # Check if any submodule is FSDP-wrapped
        for m in module.modules():
            if isinstance(m, FullyShardedDataParallel):
                return True
        # Also check for FSDP2 via GradSampleModuleFastGradientClippingFSDP
        from flashnorm.grad_sample.grad_sample_module_fast_gradient_clipping_fsdp import (
            GradSampleModuleFastGradientClippingFSDP,
        )
        if isinstance(module, GradSampleModuleFastGradientClippingFSDP):
            return True
        # Also check for FSDPFuse variant
        from flashnorm.grad_sample.grad_sample_module_fast_gradient_clipping_fsdp_fuse import (
            GradSampleModuleFastGradientClippingFSDPFuse,
        )
        return isinstance(module, GradSampleModuleFastGradientClippingFSDPFuse)
    except ImportError:
        return False


def _get_fsdp_root_module(module):
    """
    Get the root FSDP module for disabling gradient synchronization.
    
    For FSDP2 (fully_shard), returns the module with set_requires_gradient_sync() method.
    For FSDP1, returns the module with no_sync() context manager.
    For GradSampleModuleFastGradientClippingFSDP/FSDPFuse, accesses the inner _module.
    
    Args:
        module: The module (potentially a GradSampleModule wrapper)
    
    Returns:
        Tuple of (fsdp_module, api_version) where:
        - fsdp_module: The FSDP module, or None if not found
        - api_version: 'fsdp2' if set_requires_gradient_sync available, 
                      'fsdp1' if no_sync available, None otherwise
    """
    try:
        from flashnorm.grad_sample.grad_sample_module_fast_gradient_clipping_fsdp import (
            GradSampleModuleFastGradientClippingFSDP,
        )
        from flashnorm.grad_sample.grad_sample_module_fast_gradient_clipping_fsdp_fuse import (
            GradSampleModuleFastGradientClippingFSDPFuse,
        )
        
        # If it's a GradSampleModule FSDP variant, get the inner module
        if isinstance(module, (GradSampleModuleFastGradientClippingFSDP, GradSampleModuleFastGradientClippingFSDPFuse)):
            inner_module = module._module
            
            # FSDP2: Check for set_requires_gradient_sync() method
            if hasattr(inner_module, 'set_requires_gradient_sync'):
                return inner_module, 'fsdp2'
            
            # FSDP1: Check for no_sync() context manager
            if hasattr(inner_module, 'no_sync'):
                return inner_module, 'fsdp1'
        
        # Check if the module itself has FSDP methods
        if hasattr(module, 'set_requires_gradient_sync'):
            return module, 'fsdp2'
        if hasattr(module, 'no_sync'):
            return module, 'fsdp1'
            
        return None, None
    except Exception as e:
        print(f"[DEBUG] Error in _get_fsdp_root_module: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def _register_grad_blocking_hooks(module):
    """
    Register hooks that prevent gradient storage during FSDP norm pass.
    
    All parameters keep requires_grad=True so backward flows normally and
    all norm computations happen correctly. The hooks intercept gradients
    before they're stored in param.grad, preventing FSDP communication.
    
    Args:
        module: The module whose parameters need gradient blocking hooks
    
    Returns:
        List of hook handles that must be removed after norm pass
    """
    def _prevent_grad_storage_hook(grad):
        """Hook that prevents gradient from being stored in param.grad"""
        return None  # Returning None prevents storage
    
    handles = []
    for p in module.parameters():
        if p.requires_grad:
            # Register post_accumulate hook that intercepts gradient
            handle = p.register_post_accumulate_grad_hook(_prevent_grad_storage_hook)
            handles.append(handle)
    
    return handles


def _remove_grad_blocking_hooks(handles):
    """
    Remove all registered gradient blocking hooks.
    
    Args:
        handles: List of hook handles to remove
    """
    for handle in handles:
        handle.remove()


class DPTensorFastGradientClipping:
    """
    Packages the training loop for Fast Gradient and Ghost Clipping into loss.backward().
    
    Automatically detects FSDP and optimizes the norm pass by disabling parameter gradients,
    preventing expensive FSDP communication while preserving per-sample norm computation.
    
    FSDP Optimization:
    - Uses no_sync() context to prevent gradient synchronization (reduce-scatter) during norm pass
    - Uses gradient blocking hooks to prevent gradient storage and all-gather operations
    - This eliminates unnecessary FSDP communication during the first backward pass
    """

    def __init__(
        self,
        module: GradSampleModuleFastGradientClipping,
        optimizer: DPOptimizerFastGradientClipping,
        loss_per_sample: torch.Tensor,
        loss_reduction: str = "mean",
    ):
        """

        Args:
            module: the module to train
            optimizer: the optimizer used to train the module
            loss_per_sample: loss on each sample in the mini-batch of size [batch_size, 1]

        """

        self.module = module
        self.optimizer = optimizer
        self.loss_per_sample = loss_per_sample
        self.loss_reduction = loss_reduction

    def item(self):
        if self.loss_reduction == "mean":
            return torch.mean(self.loss_per_sample).detach().item()
        elif self.loss_reduction == "sum":
            return torch.sum(self.loss_per_sample).detach().item()

    def backward(self):
        """
        Repurposes loss.backward() to perform gradient clipping.
        
        - If enable_fastdp_bookkeeping=False (default): Performs two backward passes
          * With FSDP: First pass (norm pass) disables param grads to avoid FSDP comm
          * Without FSDP: Standard two-pass ghost clipping
        - If enable_fastdp_bookkeeping=True: Single pass with manual gradient computation
        """
        if self.loss_reduction == "mean":
            reduced_loss = torch.mean(self.loss_per_sample, dim=0)
        elif self.loss_reduction == "sum":
            reduced_loss = torch.sum(self.loss_per_sample, dim=0)
        else:
            raise ValueError(
                f"loss_reduction = {self.loss_reduction}. Only 'sum' and 'mean' losses are supported"
            )
        
        # Check if bookkeeping mode is enabled
        use_bookkeeping = hasattr(self.module, 'enable_fastdp_bookkeeping') and self.module.enable_fastdp_bookkeeping
        
        # Determine if FSDP is used
        is_fsdp = _is_fsdp_model(self.module)
        
        if use_bookkeeping:
            # FastDP Bookkeeping (BK) mode: Single backward pass
            # No need to retain graph since we cache intermediate values
            reduced_loss.backward()

            # Synchronize async norm computation before accessing norms
            if hasattr(self.module, 'wait_for_norms'):
                self.module.wait_for_norms()

            # Compute clipping coefficients from per-sample gradient norms
            coeff = self.module.get_clipping_coef()
    
            # Zero out any gradients from the first backward (these are non-private)
            self.optimizer.zero_grad()

            # Manually populate clipped gradients using cached activations and backprops
            # This replaces the second backward pass
            self.module.populate_clipped_gradients(coeff)

        else:
            # Two-pass ghost clipping with FSDP support

            # First backward: compute per-sample norms via hooks
            # Hooks prevent param.grad creation, reducing memory overhead
            reduced_loss.backward(retain_graph=True)

            # Zero out any gradients (should be none if FSDP, but safe to call)
            self.optimizer.zero_grad()
            
            # Deferred norm computation: compute all norms in parallel after backward
            if hasattr(self.module, 'compute_all_norms_parallel'):
                self.module.compute_all_norms_parallel()
            
            # Synchronize async norm computation before accessing norms
            if hasattr(self.module, 'wait_for_norms'):
                self.module.wait_for_norms()
            
            # Compute clipping coefficients from per-sample norms
            coeff = self.module.get_clipping_coef()

            # Second backward: compute actual parameter gradients with clipping
            second_loss_per_sample = (
                coeff.to(self.loss_per_sample.device) * self.loss_per_sample
            )
            second_loss = torch.sum(second_loss_per_sample)

            # Disable hooks to avoid recomputing norms
            self.module.disable_hooks()
            
            second_loss.backward()
            
            self.module.enable_hooks()


class DPLossFastGradientClipping:
    """
    Wrapper on the loss function to be used with Fast Gradient and Ghost Clipping. It computes the per-sample loss, and wraps it in DPTensorFastGradientClipping.
    """

    def __init__(
        self,
        module: GradSampleModuleFastGradientClipping,
        optimizer: DPOptimizerFastGradientClipping,
        criterion,
        loss_reduction: str = "mean",
    ):
        assert loss_reduction in [
            "mean",
            "sum",
        ], "loss_reduction should be either 'mean' or 'sum'"
        assert (
            loss_reduction
            == criterion.reduction
            == module.loss_reduction
            == optimizer.loss_reduction
        ), "loss_reduction should be the same across GradSampleModule, Optimizer, Criterion, and loss_reduction"

        self.optimizer = optimizer
        self.module = module
        self.criterion = criterion
        self.loss_reduction = loss_reduction
        self.criterion.reduction = "none"

    def __call__(self, *args, shape=None, **kwargs) -> DPTensorFastGradientClipping:
        """
        Redefining the forward function to compute per-sample loss and wrap it in DPTensorFastGradientClipping
        """

        loss_per_sample = self.criterion(*args, **kwargs)

        if shape is not None and loss_per_sample.shape[0] == shape[0] * shape[1]:
            # Note that the privacy unit for generative NLP tasks is per sequence.
            # The shape variable is the shape of the logits before flattening i.e., [batch_size, sequence_lenght, vocab_size].
            # This variable is necessary for ghost clipping to work with generative NLP tasks.
            loss_per_sample = loss_per_sample.view(shape[0], shape[1])  # BxT
            if self.loss_reduction == "mean":
                loss_per_sample = loss_per_sample.mean(dim=1)  # B
            elif self.loss_reduction == "sum":
                loss_per_sample = loss_per_sample.sum(dim=1)  # B
            else:
                raise ValueError(
                    f"loss_reduction = {self.loss_reduction}. Only 'sum' and 'mean' losses are supported"
                )

        return DPTensorFastGradientClipping(
            self.module, self.optimizer, loss_per_sample, self.loss_reduction
        )
