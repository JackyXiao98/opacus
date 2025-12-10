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

"""
Fused Flash Linear FSDP GradSampleModule.

This module provides a GradSampleModule that fuses per-sample gradient norm
computation directly into Linear layer backward passes, avoiding hooks for
Linear layers and eliminating FSDP serialization overhead.

Key features:
- Linear layers: Norm computed via FusedFlashLinear (no hooks)
- Non-Linear layers: Norm computed via hooks (standard approach)
- Supports both two-pass and bookkeeping (single-pass) modes
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn
from torch.distributed._tensor.experimental import implicit_replication

from opacus.grad_sample.functorch import ft_compute_per_sample_gradient
from opacus.grad_sample.fused_flash_linear import (
    TRITON_AVAILABLE,
    FusedFlashLinear,
    get_fused_linear_modules,
    replace_linear_with_fused,
)
from opacus.grad_sample.grad_sample_module_fast_gradient_clipping import (
    GradSampleModuleFastGradientClipping,
)
from opacus.utils.module_utils import requires_grad, trainable_parameters


logger = logging.getLogger(__name__)
logger.disabled = True


class GradSampleModuleFastGradientClippingFSDPFuse(GradSampleModuleFastGradientClipping):
    """
    Fused implementation of GradSampleModule for FSDP with Fast Gradient Clipping.
    
    This class replaces nn.Linear modules with FusedFlashLinear modules that
    compute per-sample gradient norms directly in the backward pass, avoiding
    the need for hooks on Linear layers.
    
    Benefits:
    - Eliminates FSDP serialization during norm computation for Linear layers
    - Reduces hook overhead for models with many Linear layers
    - Compatible with standard two-pass and bookkeeping (single-pass) modes
    - With Triton: Fuses grad_w and norm computation in single kernel pass (2x IO reduction)
    
    Note: Non-Linear layers (LayerNorm, Embedding, etc.) still use hooks.
    """
    # Note: FLASH_NORM_SAMPLERS is inherited from parent class

    def __init__(
        self,
        m: nn.Module,
        *,
        batch_first: bool = True,
        loss_reduction: str = "mean",
        strict: bool = True,
        max_grad_norm: float = 1.0,
        use_flash_clipping: bool = True,
        use_ghost_clipping: bool = True,
        enable_fastdp_bookkeeping: bool = False,
    ):
        """
        Args:
            m: nn.Module to be wrapped
            batch_first: Flag to indicate if the input tensor has batch as first dimension
            loss_reduction: "mean" or "sum" for loss aggregation
            strict: If True, validate module doesn't have buffers
            max_grad_norm: The value at which gradients are to be clipped
            use_flash_clipping: If True, use Flash Clipping for supported layers
            use_ghost_clipping: If True, use Ghost Clipping (norm-only computation)
            enable_fastdp_bookkeeping: If True, enable single-pass gradient computation
            
        Raises:
            ValueError: If enable_fastdp_bookkeeping=True but use_ghost_clipping=False
        """
        if enable_fastdp_bookkeeping and not use_ghost_clipping:
            raise ValueError(
                "enable_fastdp_bookkeeping=True requires use_ghost_clipping=True. "
                "Bookkeeping optimization only works with Ghost Clipping."
            )
        
        # Replace nn.Linear with FusedFlashLinear BEFORE calling super().__init__
        # This ensures hooks are not registered for Linear layers
        
        # Check if model already has FusedFlashLinear modules (pre-replaced)
        # or if it's FSDP-wrapped (has DTensor weights - cannot replace)
        already_has_fused = any(isinstance(mod, FusedFlashLinear) for mod in m.modules())
        is_fsdp_wrapped = self._check_fsdp_wrapped(m)
        
        if already_has_fused:
            # Model was pre-processed - skip replacement
            logger.info("Model already has FusedFlashLinear modules, skipping replacement")
        elif is_fsdp_wrapped:
            # Model is FSDP-wrapped - cannot replace safely, warn user
            import warnings
            warnings.warn(
                "Model appears to be FSDP-wrapped (has DTensor weights). "
                "Cannot replace Linear modules after FSDP wrapping. "
                "For flash_fsdp_fuse mode, replace Linear modules BEFORE FSDP wrapping. "
                "Falling back to standard hook-based norm computation for Linear layers."
            )
        else:
            # Safe to replace - model is not FSDP-wrapped and not pre-processed
            m = replace_linear_with_fused(m)
        
        # Get list of fused linear modules
        self._fused_linear_modules = get_fused_linear_modules(m)
        # Also include m itself if it's a FusedFlashLinear
        if isinstance(m, FusedFlashLinear) and m not in self._fused_linear_modules:
            self._fused_linear_modules.append(m)
        
        # Enable bookkeeping mode on fused modules if requested
        # This makes them cache activations/backprops for clipped gradient computation
        if enable_fastdp_bookkeeping:
            for module in self._fused_linear_modules:
                module.set_bookkeeping_mode(True)
        
        # Shared norm buffer for all fused linear modules
        self._linear_norm_buf: Optional[torch.Tensor] = None
        self._current_batch_size: int = 0

        # Call parent constructor
        # Note: Parent will register hooks, but FusedFlashLinear is not in GRAD_SAMPLERS
        # so no hooks will be registered for Linear layers
        super().__init__(
            m,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            strict=strict,
            force_functorch=False,
            max_grad_norm=max_grad_norm,
            use_ghost_clipping=use_ghost_clipping,
            use_flash_clipping=use_flash_clipping,
            enable_fastdp_bookkeeping=enable_fastdp_bookkeeping,
        )

    @staticmethod
    def _check_fsdp_wrapped(module: nn.Module) -> bool:
        """Check if model is FSDP-wrapped (has DTensor weights)."""
        for m in module.modules():
            if isinstance(m, nn.Linear):
                # Check if weight is a DTensor (FSDP-sharded)
                weight = m.weight
                if hasattr(weight, 'to_local') or hasattr(weight, 'full_tensor'):
                    return True
                # Also check type name for DTensor
                if 'DTensor' in str(type(weight)):
                    return True
        return False

    def _get_module_type(self, module: nn.Module) -> type:
        """Get the actual module type, handling FSDP wrapped modules."""
        module_type = (
            module.__class__.__bases__[1]
            if isinstance(module, torch.distributed.fsdp.FSDPModule)
            else type(module)
        )
        return module_type

    def _setup_norm_buffer(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """
        Set up the shared norm buffer for fused Linear modules.
        
        This should be called before each forward pass with the current batch size.
        
        Args:
            batch_size: Current batch size
            device: Device for the buffer
            dtype: Data type for the buffer (will be converted to float if integer)
        """
        # Norm buffer must be floating point for accumulating squared norms
        # Input dtype might be integer (e.g., input_ids), so we need to convert
        if not dtype.is_floating_point:
            dtype = torch.float32
        
        # Create or resize buffer if needed
        if self._linear_norm_buf is None or self._current_batch_size != batch_size:
            self._linear_norm_buf = torch.zeros(batch_size, device=device, dtype=dtype)
            self._current_batch_size = batch_size
        else:
            # Zero out existing buffer
            self._linear_norm_buf.zero_()
        
        # Set buffer on all fused linear modules
        for module in self._fused_linear_modules:
            module.set_norm_buffer(self._linear_norm_buf)

    def _enable_fused_norm_computation(self, enable: bool = True):
        """Enable or disable norm computation in fused Linear modules."""
        for module in self._fused_linear_modules:
            module.set_compute_norms(enable)

    def forward(self, *args, **kwargs):
        """
        Forward pass with automatic norm buffer setup.
        
        Detects batch size from input and sets up norm buffer accordingly.
        """
        # Detect batch size from first tensor argument
        batch_size = None
        device = None
        dtype = None
        
        for arg in args:
            if isinstance(arg, torch.Tensor):
                batch_size = arg.shape[0]
                device = arg.device
                dtype = arg.dtype
                break
        
        if batch_size is None:
            for v in kwargs.values():
                if isinstance(v, torch.Tensor):
                    batch_size = v.shape[0]
                    device = v.device
                    dtype = v.dtype
                    break
        
        if batch_size is not None and self.hooks_enabled:
            self._setup_norm_buffer(batch_size, device, dtype)
            self._enable_fused_norm_computation(True)
        
        return self._module(*args, **kwargs)

    def get_norm_sample(self) -> torch.Tensor:
        """
        Get per-example gradient norms combining fused Linear norms and hook-based norms.
        
        Returns:
            Per-sample gradient norms [batch_size]
        """
        # Collect norms from hooked layers (non-Linear)
        hooked_norms = []
        for module in self.iterate_submodules(self._module):
            if hasattr(module, 'norm_sample') and module.norm_sample:
                for per_param_norm in module.norm_sample:
                    hooked_norms.append(per_param_norm)
        
        # Compute squared norm sum from hooked layers
        if hooked_norms:
            stacked_hooked = torch.stack(hooked_norms, dim=0)
            hooked_norm_squared = (stacked_hooked ** 2).sum(dim=0)
        else:
            hooked_norm_squared = torch.zeros(
                self._current_batch_size, 
                device=self._linear_norm_buf.device if self._linear_norm_buf is not None else 'cpu',
                dtype=self._linear_norm_buf.dtype if self._linear_norm_buf is not None else torch.float32,
            )
        
        # Get squared norms from fused Linear layers (already accumulated in buffer)
        if self._linear_norm_buf is not None:
            linear_norm_squared = self._linear_norm_buf
        else:
            linear_norm_squared = torch.zeros_like(hooked_norm_squared)
        
        # Combine: total_norm^2 = linear_norm^2 + hooked_norm^2
        total_norm_squared = linear_norm_squared + hooked_norm_squared.squeeze(-1)
        
        # All-reduce for FSDP (sum squared norms across ranks)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(total_norm_squared, op=torch.distributed.ReduceOp.SUM)
        
        # Take sqrt to get final norms
        norm_sample = torch.sqrt(total_norm_squared + 1e-12)
        
        self.per_sample_gradient_norms = norm_sample
        return norm_sample

    def capture_activations_hook(
        self,
        module: nn.Module,
        forward_input: List[torch.Tensor],
        _forward_output: torch.Tensor,
    ):
        """
        Capture activations hook - skips FusedFlashLinear modules.
        
        FusedFlashLinear modules compute norms directly in backward,
        so we don't need to capture activations for them.
        """
        # Skip FusedFlashLinear - it handles its own norm computation
        if isinstance(module, FusedFlashLinear):
            return
        
        if (
            not requires_grad(module)
            or not module.training
            or not torch.is_grad_enabled()
            or not self.hooks_enabled
        ):
            return

        if not hasattr(module, "activations"):
            module.activations = []
        module.activations.append([t.detach() for t in forward_input])

        if not hasattr(module, "_forward_counter"):
            module._forward_counter = 0

        module._forward_counter += 1
        if self.use_ghost_clipping and module._forward_counter > 1:
            raise NotImplementedError("Parameter tying is not supported with FSDP Fuse")

    def capture_backprops_hook(
        self,
        module: nn.Module,
        _forward_input: torch.Tensor,
        forward_output: torch.Tensor,
        loss_reduction: str,
        batch_first: bool,
    ):
        """
        Capture backprops hook - skips FusedFlashLinear modules.
        
        For non-Linear layers, computes per-sample gradient norms using hooks.
        FusedFlashLinear modules handle norm computation in their backward pass.
        """
        # Skip FusedFlashLinear - it handles its own norm computation
        if isinstance(module, FusedFlashLinear):
            return
        
        if not self.hooks_enabled:
            return

        backprops = forward_output[0].detach()

        activations, backprops = self.rearrange_grad_samples(
            module=module,
            backprops=backprops,
            loss_reduction=loss_reduction,
            batch_first=batch_first,
        )
        
        module_type = self._get_module_type(module)
        module._forward_counter -= 1
        
        # Initialize norm_sample storage
        if not hasattr(module, "norm_sample"):
            module.norm_sample = []
            for _, param in trainable_parameters(module):
                module.norm_sample.append(
                    torch.zeros(
                        torch.Size([module.max_batch_len, 1]),
                        device=param.device,
                        dtype=param.dtype,
                    )
                )

        if self.use_ghost_clipping and (
            module_type in self.NORM_SAMPLERS or 
            (self.use_flash_clipping and module_type in self.FLASH_NORM_SAMPLERS)
        ):
            # Use Flash sampler if available and enabled
            if self.use_flash_clipping and module_type in self.FLASH_NORM_SAMPLERS:
                norm_sampler_fn = self.FLASH_NORM_SAMPLERS[module_type]
            else:
                norm_sampler_fn = self.NORM_SAMPLERS[module_type]
            
            norm_samples = norm_sampler_fn(module, activations, backprops)

            for idx, (param, ns) in enumerate(
                (item for item in norm_samples.items() if item[0].requires_grad)
            ):
                if idx < len(module.norm_sample):
                    module.norm_sample[idx] = ns
            
            # FastDP Bookkeeping: Cache activations and backprops
            if self.enable_fastdp_bookkeeping:
                self._bk_cache.append({
                    'module': module,
                    'activations': activations,
                    'backprops': backprops,
                })
        else:
            # Use grad sampler for layers without norm sampler
            if not self.force_functorch and module_type in self.GRAD_SAMPLERS:
                grad_sampler_fn = self.GRAD_SAMPLERS[module_type]
            else:
                grad_sampler_fn = ft_compute_per_sample_gradient

            grad_samples = grad_sampler_fn(module, activations, backprops)

            for idx, (_, gs) in enumerate((item for item in grad_samples.items())):
                module.norm_sample[idx] = gs.reshape(len(gs), -1).norm(2, dim=-1)
            del grad_samples

        if len(module.activations) == 0:
            if hasattr(module, "max_batch_len"):
                del module.max_batch_len

    def populate_clipped_gradients(self, clipping_coef: torch.Tensor):
        """
        Manually compute clipped gradients using cached data.
        
        For fused Linear layers, gradients are computed directly from activations
        and backprops. For hooked layers, uses the standard bookkeeping approach.
        
        Args:
            clipping_coef: Per-sample clipping coefficients [batch_size]
        """
        if not self.enable_fastdp_bookkeeping:
            raise RuntimeError(
                "populate_clipped_gradients() requires enable_fastdp_bookkeeping=True"
            )
        
        # Use implicit_replication for FSDP DTensor compatibility
        with implicit_replication():
            self._populate_gradients_impl(clipping_coef)
    
    def _populate_gradients_impl(self, clipping_coef: torch.Tensor):
        """
        Internal implementation of gradient population.
        
        Handles both fused Linear layers and hooked layers.
        
        Optimizations:
        - Uses clipped_backprops directly (avoids double clipping)
        - Pre-computes coef_with_scale for fused modules
        - Uses in-place add_() operations
        - Unified handling for all layer types (no code duplication)
        """
        # --- Part 1: Populate gradients for hooked layers ---
        if self._bk_cache is not None and len(self._bk_cache) > 0:
            for cache_entry in self._bk_cache:
                module = cache_entry['module']
                activations = cache_entry['activations']
                backprops = cache_entry['backprops']
                
                # Apply per-sample clipping coefficients to backprops ONCE
                # Support arbitrary dimensions: 2D [B, d], 3D [B, T, d], 4D [B, C, H, W], etc.
                coef_shape = [-1] + [1] * (backprops.dim() - 1)
                coef_on_device = clipping_coef.view(*coef_shape).to(
                    device=backprops.device, dtype=backprops.dtype
                )
                clipped_backprops = backprops * coef_on_device
                
                module_type = self._get_module_type(module)
                
                # Compute gradients using CLIPPED backprops (fix: was using original backprops)
                if not self.force_functorch and module_type in self.GRAD_SAMPLERS:
                    grad_sampler_fn = self.GRAD_SAMPLERS[module_type]
                else:
                    grad_sampler_fn = ft_compute_per_sample_gradient
                
                # Use clipped_backprops - gradients are already scaled by clipping_coef
                grad_samples = grad_sampler_fn(module, activations, clipped_backprops)
                
                # Sum over batch dimension - no need to clip again since backprops were clipped
                for param, gs in grad_samples.items():
                    if param.requires_grad:
                        grad = gs.sum(dim=0)
                        
                        if param.grad is None:
                            param.grad = grad
                        else:
                            param.grad.add_(grad)  # in-place add
                
                # Free memory immediately
                del cache_entry, module, activations, backprops, clipped_backprops
            
            self._bk_cache.clear()
        
        # --- Part 2: Handle fused Linear layers in BK mode ---
        # Pre-compute coef_with_scale to avoid repeated computation in each module
        # When loss_reduction="mean", grad_out is divided by batch_size, so we need
        # to multiply by batch_size to get correct gradient magnitude
        grad_scale = float(self._current_batch_size) if self.loss_reduction == "mean" else 1.0
        
        # Collect modules with cache first to avoid repeated checks
        modules_with_cache = [m for m in self._fused_linear_modules if m._bk_cache is not None]
        
        if modules_with_cache:
            # Pre-multiply clipping_coef with grad_scale once
            if grad_scale != 1.0:
                coef_with_scale = clipping_coef * grad_scale
            else:
                coef_with_scale = clipping_coef
            
            # Process each module and immediately clear its cache to minimize peak memory
            # This ensures that only one layer's cache exists at a time during gradient computation
            for module in modules_with_cache:
                # Pass pre-scaled coef, grad_scale=1.0 since already scaled
                module.compute_clipped_gradient(coef_with_scale, grad_scale=1.0)
                # Immediately clear cache after computing gradients to free memory
                # This is critical for models with many layers to avoid O(num_layers × B × T × D) memory
                module.clear_bk_cache()

    def disable_hooks(self):
        """Disable hooks and norm computation for second backward pass."""
        super().disable_hooks()
        self._enable_fused_norm_computation(False)
    
    def enable_hooks(self):
        """Enable hooks and norm computation."""
        super().enable_hooks()
        self._enable_fused_norm_computation(True)

    def clear_bookkeeping_cache(self):
        """Clear the bookkeeping cache to free memory."""
        if self._bk_cache is not None:
            self._bk_cache.clear()
        # Also clear fused module caches
        for module in self._fused_linear_modules:
            module.clear_bk_cache()

