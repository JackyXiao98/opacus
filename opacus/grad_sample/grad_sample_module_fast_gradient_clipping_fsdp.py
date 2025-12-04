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

from __future__ import annotations

import logging
import os
from typing import List

import torch
import torch.nn as nn
from torch.distributed._tensor.experimental import implicit_replication
from opacus.grad_sample.functorch import ft_compute_per_sample_gradient
from opacus.grad_sample.grad_sample_module_fast_gradient_clipping import (
    GradSampleModuleFastGradientClipping,
)
from opacus.utils.module_utils import requires_grad, trainable_parameters


logger = logging.getLogger(__name__)
logger.disabled = True


class GradSampleModuleFastGradientClippingFSDP(GradSampleModuleFastGradientClipping):
    """
    Hooks-based implementation of GradSampleModule with Fast Gradient and Ghost Clipping and FSDP support

    Computes norms of gradients without gradient instantiation
    """
    # Note: FLASH_NORM_SAMPLERS is inherited from parent class
    # Do not redefine it here, otherwise it will create an empty dict that shadows the parent's registered samplers

    def __init__(
        self,
        m: nn.Module,
        *,
        batch_first: bool = True,
        loss_reduction="mean",
        strict: bool = True,
        max_grad_norm=1,
        use_flash_clipping=False,
        use_ghost_clipping=True,
        enable_fastdp_bookkeeping=False,
    ):
        """

        Args:
            m: nn.Module to be wrapped
            batch_first: Flag to indicate if the input tensor to the corresponding module
                has the first dimension representing the batch. If set to True, dimensions on
                input tensor are expected be ``[batch_size, ...]``, otherwise
                ``[K, batch_size, ...]``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            max_grad_norm: The value at which gradients are to be clipped.
            strict: If set to True, the input module will be validated to make sure that
                it does not have buffers in all its submodules.
            enable_fastdp_bookkeeping: If set to ``True``, enables FastDP Bookkeeping (BK)
                optimization which caches activations and backprops to enable single-pass
                gradient clipping instead of two backward passes. This reduces memory overhead
                from gradient graph retention. Only compatible with use_ghost_clipping=True.

        Raises:
            NotImplementedError
                If ``strict`` is set to ``True`` and module ``m`` (or any of its
                submodules) includes a buffer.
            ValueError
                If ``enable_fastdp_bookkeeping`` is True but ``use_ghost_clipping`` is False.
        """
        if enable_fastdp_bookkeeping and not use_ghost_clipping:
            raise ValueError(
                "enable_fastdp_bookkeeping=True requires use_ghost_clipping=True. "
                "Bookkeeping optimization only works with Ghost Clipping."
            )

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

        # Deferred norm computation cache
        self._deferred_norm_cache = []
        self._use_deferred_norm = os.environ.get('OPACUS_USE_DEFERRED_NORM', '0') == '1'

    def _get_module_type(self, module: nn.Module) -> str:
        module_type = (
            module.__class__.__bases__[1]
            if isinstance(module, torch.distributed.fsdp.FSDPModule)
            else type(module)
        )
        return module_type

    def get_norm_sample(self) -> torch.Tensor:
        """
        Get per-example gradient norms with distributed reduction.
        
        This is different from the parent class as norm_sample is an attribute of the module 
        instead of the parameter. For FSDP, we need to all-reduce the per-sample norms across 
        ranks to get the global gradient norms.
        """
        # Stack per-parameter norms from all modules
        stacked_norms = torch.stack(
            [
                per_param_norm
                for module in self.iterate_submodules(self._module)
                for per_param_norm in module.norm_sample
            ],
            dim=0,
        )
        
        # Compute local contribution: sum of squared norms
        # norm^2 = sum(param_i_norm^2) for parameters on this rank
        norm_sample_squared = (stacked_norms ** 2).sum(dim=0)

        # All-reduce the squared norms across ranks to get global per-sample gradient norms
        # This is critical for FSDP: we need sqrt(sum_all_ranks(norm_i^2)), not sum(sqrt(norm_i^2))
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(norm_sample_squared, op=torch.distributed.ReduceOp.SUM)

        # Take square root to get final per-sample gradient norms
        norm_sample = torch.sqrt(norm_sample_squared + 1e-12)  # Add epsilon for numerical stability

        self.per_sample_gradient_norms = norm_sample
        return norm_sample

    def capture_activations_hook(
        self,
        module: nn.Module,
        forward_input: List[torch.Tensor],
        _forward_output: torch.Tensor,
    ):
        """Captures activations for the given module.
        This function is similar to the capture_activations_hook in the parent class (GradSampleModuleFastGradientClipping),
        except that it attaches _forward_counter to the module instead of parameter variable.
        Another difference is that GradSampleModuleFastGradientClipping doesn't support tied parameters only under Ghost Clipping,
        but this class doesn't supports tied parameters for either Fast Gradient Clipping or Ghost Clipping.
        """
        if (
            not requires_grad(module)
            or not module.training
            or not torch.is_grad_enabled()
            or not self.hooks_enabled
        ):
            return

        if not hasattr(module, "activations"):
            module.activations = []
        module.activations.append([t.detach() for t in forward_input])  # pyre-ignore

        if not hasattr(module, "_forward_counter"):
            module._forward_counter = 0

        module._forward_counter += 1
        if self.use_ghost_clipping and module._forward_counter > 1:
            raise NotImplementedError("Parameter tying is not supported with FSDP")

    def capture_backprops_hook(
        self,
        module: nn.Module,
        _forward_input: torch.Tensor,
        forward_output: torch.Tensor,
        loss_reduction: str,
        batch_first: bool,
    ):
        """
        Computes norms of per sample gradient given the current backprops and activations
        stored by the associated forward hook. Computed per sample gradient norms are
        stored in ``norm_sample`` field in each module.
        This function differs from capture_backprops_hook in GradSampleModuleFastGradientClipping in that
        it attaches all the attributes to the module instead of the parameter variable.

        Args:
            module: nn.Module,
            _forward_input: torch.Tensor,
            forward_output: torch.Tensor,
            loss_reduction: str,
            batch_first: bool,
        """
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
        
        # Deferred norm computation mode: only collect data, don't compute norms
        if self._use_deferred_norm:
            # IMPORTANT: In Ghost Clipping mode, gradient sync is disabled during first backward
            # Calling trainable_parameters() here would trigger FSDP all-gather and cause deadlock
            # Solution: Don't count parameters here, defer everything to compute_all_norms_parallel()
            
            # Cache metadata before module state gets cleaned up
            cached_max_batch_len = module.max_batch_len if hasattr(module, "max_batch_len") else None
            
            # Mark norm_sample as uninitialized (will be created later)
            if not hasattr(module, "norm_sample"):
                module.norm_sample = None
            
            # Cache data for later parallel computation
            # DO NOT call trainable_parameters() here to avoid FSDP deadlock!
            self._deferred_norm_cache.append({
                'module': module,
                'module_type': module_type,
                'activations': activations,
                'backprops': backprops,
                'loss_reduction': loss_reduction,
                'batch_first': batch_first,
                'max_batch_len': cached_max_batch_len,
            })
            
            # Still cache for bookkeeping if enabled
            if self.enable_fastdp_bookkeeping:
                self._bk_cache.append({
                    'module': module,
                    'activations': activations,
                    'backprops': backprops,
                })
            
            # Exit early - no immediate norm computation
            if len(module.activations) == 0:
                if hasattr(module, "max_batch_len"):
                    del module.max_batch_len
            return

        # Original immediate computation path
        if not hasattr(module, "norm_sample"):
            # currently, we don't support freezing and unfreezing params in between training. Making this a dictionary and mapping with param names might fix this.
            module.norm_sample = []
            # This call to trainable_parameters triggers FSDP all-gather!
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
            # Use Flash sampler if available and enabled, otherwise use standard sampler
            if self.use_flash_clipping and module_type in self.FLASH_NORM_SAMPLERS:
                norm_sampler_fn = self.FLASH_NORM_SAMPLERS[module_type]
            else:
                norm_sampler_fn = self.NORM_SAMPLERS[module_type]
            
            norm_samples = norm_sampler_fn(module, activations, backprops)

            for idx, (_, ns) in enumerate(
                (item for item in norm_samples.items() if item[0].requires_grad)
            ):
                module.norm_sample[idx] = ns
            
            # FastDP Bookkeeping: Cache activations and backprops for later gradient computation
            if self.enable_fastdp_bookkeeping:
                # Store direct references (already detached, no clone needed to save memory)
                self._bk_cache.append({
                    'module': module,
                    'activations': activations,
                    'backprops': backprops,
                })
        else:
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
        Manually compute clipped gradients using cached activations and backprops.
        This implements the Bookkeeping (BK) algorithm from FastDP for FSDP.
        
        This method should be called after computing clipping coefficients and only
        when enable_fastdp_bookkeeping=True. It uses cached intermediate values
        from the forward/backward pass to directly compute clipped gradients without
        a second backward pass.
        
        Args:
            clipping_coef: Per-sample clipping coefficients [batch_size], where
                          clipping_coef[i] = min(1, max_grad_norm / ||grad_i||)
        """
        if not self.enable_fastdp_bookkeeping:
            raise RuntimeError(
                "populate_clipped_gradients() requires enable_fastdp_bookkeeping=True"
            )
        
        if self._bk_cache is None or len(self._bk_cache) == 0:
            raise RuntimeError(
                "Bookkeeping cache is empty. Make sure to call forward and backward first."
            )
        
        # Use implicit_replication context manager to handle DTensor conversion for FSDP
        # This allows regular torch.Tensors to be automatically converted to DTensors
        # when assigning to FSDP parameters
        with implicit_replication():
            # Process each cached layer one-by-one with immediate cleanup to minimize peak memory
            self._populate_gradients_impl(clipping_coef)
    
    def _populate_gradients_impl(self, clipping_coef: torch.Tensor):
        """
        Internal implementation of gradient population.
        Separated to allow wrapping with implicit_replication context manager.
        """
        for i in range(len(self._bk_cache)):
            cache_entry = self._bk_cache[i]
            module = cache_entry['module']
            activations = cache_entry['activations']
            backprops = cache_entry['backprops']
            
            # Apply per-sample clipping coefficients to backprops
            # backprops shape: [batch_size, ...], clipping_coef shape: [batch_size]
            # We need to reshape clipping_coef to broadcast properly and match dtype
            # Support arbitrary dimensions: 2D [B, d], 3D [B, T, d], 4D [B, C, H, W], etc.
            coef_shape = [-1] + [1] * (backprops.dim() - 1)
            clipped_backprops = backprops * clipping_coef.view(*coef_shape).to(device=backprops.device, dtype=backprops.dtype)
            
            module_type = self._get_module_type(module)
            
            # Now compute gradients using the clipped backprops
            # This is equivalent to the second backward pass but done manually
            if module_type == nn.Linear:
                A = activations[0]
                
                for idx, (param_name, param) in enumerate(trainable_parameters(module)):
                    if param_name.endswith('weight') and param.requires_grad:
                        # Gradient: sum over batch of outer products
                        # clipped_backprops: [B, d_out] or [B, T, d_out]
                        # A: [B, d_in] or [B, T, d_in]
                        if clipped_backprops.dim() == 2:
                            # Standard case: [B, d_out] x [B, d_in] -> sum over B
                            grad_weight = torch.einsum("bi,bj->ij", clipped_backprops, A)
                        else:
                            # Sequence case: [B, T, d_out] x [B, T, d_in] -> sum over B and T
                            grad_weight = torch.einsum("bti,btj->ij", clipped_backprops, A)
                        
                        # Accumulate into .grad (in case there are multiple batches)
                        if param.grad is None:
                            param.grad = grad_weight
                        else:
                            param.grad += grad_weight
                    
                    elif param_name.endswith('bias') and param.requires_grad:
                        # Bias gradient: sum over batch (and time if 3D)
                        if clipped_backprops.dim() == 2:
                            grad_bias = torch.einsum("bi->i", clipped_backprops)
                        else:
                            grad_bias = torch.einsum("bti->i", clipped_backprops)
                        
                        if param.grad is None:
                            param.grad = grad_bias
                        else:
                            param.grad += grad_bias
            
            elif module_type == nn.LayerNorm:
                # For LayerNorm, we need to use the grad_sampler function
                # Get the grad sampler function
                if not self.force_functorch and module_type in self.GRAD_SAMPLERS:
                    grad_sampler_fn = self.GRAD_SAMPLERS[module_type]
                else:
                    grad_sampler_fn = ft_compute_per_sample_gradient
                
                # Compute per-sample gradients
                grad_samples = grad_sampler_fn(module, activations, backprops)
                
                # Apply clipping coefficients and sum
                for param, gs in grad_samples.items():
                    if param.requires_grad:
                        # gs shape: [B, ...] for per-sample gradients
                        # Apply clipping coefficient: [B] -> [B, 1, 1, ...]
                        coef_shape = [gs.shape[0]] + [1] * (gs.dim() - 1)
                        clipped_gs = gs * clipping_coef.view(*coef_shape).to(device=gs.device, dtype=gs.dtype)
                        
                        # Sum over batch dimension
                        grad = clipped_gs.sum(dim=0)
                        
                        if param.grad is None:
                            param.grad = grad
                        else:
                            param.grad += grad
            
            elif module_type == nn.Embedding:
                # For Embedding, we need to handle sparse gradients
                if not self.force_functorch and module_type in self.GRAD_SAMPLERS:
                    grad_sampler_fn = self.GRAD_SAMPLERS[module_type]
                else:
                    grad_sampler_fn = ft_compute_per_sample_gradient
                
                grad_samples = grad_sampler_fn(module, activations, backprops)
                
                for param, gs in grad_samples.items():
                    if param.requires_grad:
                        coef_shape = [gs.shape[0]] + [1] * (gs.dim() - 1)
                        clipped_gs = gs * clipping_coef.view(*coef_shape).to(device=gs.device, dtype=gs.dtype)
                        grad = clipped_gs.sum(dim=0)
                        
                        if param.grad is None:
                            param.grad = grad
                        else:
                            param.grad += grad
            
            else:
                # For other layer types, use the registered grad_sampler
                if not self.force_functorch and module_type in self.GRAD_SAMPLERS:
                    grad_sampler_fn = self.GRAD_SAMPLERS[module_type]
                else:
                    grad_sampler_fn = ft_compute_per_sample_gradient
                
                grad_samples = grad_sampler_fn(module, activations, backprops)
                
                for param, gs in grad_samples.items():
                    if param.requires_grad:
                        coef_shape = [gs.shape[0]] + [1] * (gs.dim() - 1)
                        clipped_gs = gs * clipping_coef.view(*coef_shape).to(device=gs.device, dtype=gs.dtype)
                        grad = clipped_gs.sum(dim=0)
                        
                        if param.grad is None:
                            param.grad = grad
                        else:
                            param.grad += grad
            
            # Immediately free memory for this cache entry to minimize peak memory
            self._bk_cache[i] = None
            del cache_entry, module, activations, backprops
        
        # Clear cache list after use to free memory
        self._bk_cache.clear()
    
    def clear_bookkeeping_cache(self):
        """Clear the bookkeeping cache to free memory."""
        if self._bk_cache is not None:
            self._bk_cache.clear()
    
    def compute_all_norms_parallel(self):
        """
        Compute all deferred norms in parallel after backward pass completes.
        This is the key optimization that replaces per-layer serial norm computation.
        
        Called after the first backward pass when using deferred norm computation mode.
        """
        if not hasattr(self, '_deferred_norm_cache') or len(self._deferred_norm_cache) == 0:
            return
        
        # Process all cached layers and compute norms
        for layer_data in self._deferred_norm_cache:
            self._compute_single_layer_norm(layer_data)
        
        # Clear cache after computation
        self._deferred_norm_cache.clear()
    
    def _compute_single_layer_norm(self, layer_data):
        """
        Compute norm for a single layer using cached activations and backprops.
        This is called by compute_all_norms_parallel() for each layer.
        
        Args:
            layer_data: Dict containing module, activations, backprops, and metadata
        """
        module = layer_data['module']
        module_type = layer_data['module_type']
        activations = layer_data['activations']
        backprops = layer_data['backprops']
        max_batch_len = layer_data.get('max_batch_len')
        
        # Initialize norm_sample storage if needed
        # In deferred mode, this was set to None, so we need to properly initialize it
        # This is called AFTER backward completes and gradient sync is re-enabled,
        # so it's now safe to call trainable_parameters()
        if not hasattr(module, "norm_sample") or module.norm_sample is None:
            module.norm_sample = []
            
            # Infer device and dtype from backprops instead of accessing parameters
            device = backprops.device
            dtype = backprops.dtype
            
            # Now it's safe to count parameters (gradient sync is re-enabled)
            num_params = sum(1 for _ in trainable_parameters(module))
            
            # Create norm_sample tensors
            for _ in range(num_params):
                module.norm_sample.append(
                    torch.zeros(
                        torch.Size([max_batch_len, 1]),
                        device=device,
                        dtype=dtype,
                    )
                )
        
        # Compute norms using appropriate sampler
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
            
            # IMPORTANT: In deferred mode, avoid accessing parameter attributes
            # The norm_sampler returns a dict {param: norm_tensor}
            # We only care about the norm values, in the order they were created
            # Since trainable_parameters() returns params in a consistent order,
            # and norm_sampler uses the same iteration, we can just use values()
            for idx, ns in enumerate(norm_samples.values()):
                if idx < len(module.norm_sample):
                    module.norm_sample[idx] = ns
        else:
            # Use grad sampler for non-ghost clipping layers
            if not self.force_functorch and module_type in self.GRAD_SAMPLERS:
                grad_sampler_fn = self.GRAD_SAMPLERS[module_type]
                grad_samples = grad_sampler_fn(module, activations, backprops)
                
                for idx, (_, gs) in enumerate((item for item in grad_samples.items())):
                    module.norm_sample[idx] = gs.reshape(len(gs), -1).norm(2, dim=-1)
                del grad_samples
            else:
                # functorch doesn't work with FSDP DTensors in deferred norm mode
                # For layers without specialized samplers, check if they have trainable parameters
                has_trainable_params = any(p.requires_grad for p in module.parameters(recurse=False))
                
                if has_trainable_params:
                    # Layer has trainable parameters but no specialized sampler
                    # This will lead to incorrect gradient norms, so we must error out
                    raise NotImplementedError(
                        f"Layer type {module_type.__name__} doesn't have a specialized grad_sampler "
                        f"and cannot be used with FSDP + deferred norm computation. "
                        f"Please register a grad_sampler or norm_sampler for this layer type, "
                        f"or disable deferred norm computation (set OPACUS_USE_DEFERRED_NORM=0)."
                    )
                else:
                    # Layer has no trainable parameters, safe to skip (contributes zero to grad norm)
                    # This should not happen as we filter in capture_backprops_hook, but handle gracefully
                    pass
    
    def verify_norm_correctness(self, rtol=1e-4, atol=1e-6):
        """
        Verify that deferred norm computation produces the same results as immediate computation.
        This should be called after both methods have computed norms for comparison.
        
        Args:
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison
            
        Returns:
            bool: True if norms match within tolerance, raises ValueError otherwise
        """
        if not hasattr(self, '_deferred_norms') or not hasattr(self, '_immediate_norms'):
            raise RuntimeError("Both deferred and immediate norms must be computed before verification")
        
        deferred = self._deferred_norms
        immediate = self._immediate_norms
        
        if len(deferred) != len(immediate):
            raise ValueError(
                f"Norm count mismatch: deferred={len(deferred)}, immediate={len(immediate)}"
            )
        
        max_diff = 0.0
        max_rel_diff = 0.0
        
        for i, (d_norm, i_norm) in enumerate(zip(deferred, immediate)):
            if not torch.allclose(d_norm, i_norm, rtol=rtol, atol=atol):
                diff = (d_norm - i_norm).abs()
                max_diff_i = diff.max().item()
                rel_diff = (diff / (i_norm.abs() + 1e-10)).max().item()
                
                if max_diff_i > max_diff:
                    max_diff = max_diff_i
                if rel_diff > max_rel_diff:
                    max_rel_diff = rel_diff
        
        if max_diff > atol or max_rel_diff > rtol:
            raise ValueError(
                f"Norm mismatch detected!\n"
                f"  Max absolute difference: {max_diff:.6e}\n"
                f"  Max relative difference: {max_rel_diff:.6e}\n"
                f"  Tolerance: rtol={rtol}, atol={atol}"
            )
        
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            print(f"\n✓ Norm correctness verified!")
            print(f"  Compared {len(deferred)} norms")
            print(f"  Max absolute difference: {max_diff:.6e}")
            print(f"  Max relative difference: {max_rel_diff:.6e}\n")
        
        return True
