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


class GradSampleModuleFastGradientClippingFSDPAsync(GradSampleModuleFastGradientClipping):
    """
    Async CUDA stream version of GradSampleModuleFastGradientClippingFSDP
    
    Uses async CUDA streams to overlap norm computation with FSDP communication,
    eliminating pipeline bubbles during backward pass.
    
    Key optimizations:
    - Norm computation happens in parallel CUDA stream
    - No blocking of FSDP backward communication
    - Memory-safe tensor lifetime management
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

        # Async CUDA stream for norm computation
        self._norm_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self._async_keep_alive = []  # Keep tensors alive during async computation
        self._async_norm_futures = []  # Store pending norm results

    def _get_module_type(self, module: nn.Module) -> str:
        module_type = (
            module.__class__.__bases__[1]
            if isinstance(module, torch.distributed.fsdp.FSDPModule)
            else type(module)
        )
        return module_type

    def _is_fsdp_active(self) -> bool:
        """Check if model is actually using FSDP."""
        try:
            # Check if any module is FSDP wrapped
            for m in self._module.modules():
                if hasattr(m, '__class__'):
                    class_name = m.__class__.__name__
                    type_str = str(type(m)).lower()
                    if 'FSDP' in class_name or 'fsdp' in type_str:
                        return True
            return False
        except:
            return False

    def wait_for_norms(self):
        """
        Synchronize async norm stream before accessing norms.
        
        This must be called before:
        - Accessing per_sample_gradient_norms
        - Computing clipping coefficients
        - Performing optimizer step
        """
        if self._norm_stream is not None:
            torch.cuda.current_stream().wait_stream(self._norm_stream)
        
        # Clear cached tensors after sync
        self._async_keep_alive.clear()
        self._async_norm_futures.clear()

    def get_norm_sample(self) -> torch.Tensor:
        """
        Get per-example gradient norms with distributed reduction.
        
        Automatically synchronizes async norm computation before accessing norms.
        """
        # Wait for async norm computation to complete
        self.wait_for_norms()
        
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
        Computes norms of per sample gradient using async CUDA streams.
        
        Key differences from synchronous version:
        - Parameter checks happen in main thread before stream context
        - Norm computation launches in async stream
        - Returns immediately without blocking
        - Tensors kept alive in _async_keep_alive
        
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
        
        # MAIN THREAD: Initialize norm_sample storage and check parameter requirements
        # This must happen BEFORE entering the async stream context
        if not hasattr(module, "norm_sample"):
            module.norm_sample = []
            # Safe to call trainable_parameters here (main thread, no stream context)
            for _, param in trainable_parameters(module):
                module.norm_sample.append(
                    torch.zeros(
                        torch.Size([module.max_batch_len, 1]),
                        device=param.device,
                        dtype=param.dtype,
                    )
                )
        
        # MAIN THREAD: Check parameter requirements (for Linear layers with flash clipping)
        weight_requires_grad = False
        bias_requires_grad = False
        weight_param = None
        bias_param = None
        
        if module_type == nn.Linear and self.use_flash_clipping:
            # Check parameters in main thread before entering async stream
            for param_name, param in trainable_parameters(module):
                if param_name.endswith('weight'):
                    weight_requires_grad = param.requires_grad
                    weight_param = param
                elif param_name.endswith('bias'):
                    bias_requires_grad = param.requires_grad
                    bias_param = param

        # ASYNC STREAM: Launch norm computation without blocking (only if FSDP is active)
        # Only use async stream if FSDP is active (otherwise just adds overhead)
        use_async = self._norm_stream is not None and torch.cuda.is_available() and self._is_fsdp_active()
        
        if use_async:
            # Wait for main stream to ensure activations and backprops are ready
            self._norm_stream.wait_stream(torch.cuda.current_stream())
            
            with torch.cuda.stream(self._norm_stream):
                self._compute_norms_async(
                    module=module,
                    module_type=module_type,
                    activations=activations,
                    backprops=backprops,
                    weight_requires_grad=weight_requires_grad,
                    bias_requires_grad=bias_requires_grad,
                    weight_param=weight_param,
                    bias_param=bias_param,
                )
            
            # Keep tensors alive until async computation completes
            self._async_keep_alive.append({
                'activations': activations,
                'backprops': backprops,
                'module': module,
            })
            
            # Prevent unbounded growth - if cache too large, sync immediately
            if len(self._async_keep_alive) > 100:  # Safety limit
                self.wait_for_norms()
        else:
            # CPU fallback or non-FSDP: compute synchronously
            self._compute_norms_async(
                module=module,
                module_type=module_type,
                activations=activations,
                backprops=backprops,
                weight_requires_grad=weight_requires_grad,
                bias_requires_grad=bias_requires_grad,
                weight_param=weight_param,
                bias_param=bias_param,
            )
        
        # Clean up activations
        if len(module.activations) == 0:
            if hasattr(module, "max_batch_len"):
                del module.max_batch_len

    def _compute_norms_async(
        self,
        module: nn.Module,
        module_type: type,
        activations: List[torch.Tensor],
        backprops: torch.Tensor,
        weight_requires_grad: bool,
        bias_requires_grad: bool,
        weight_param: nn.Parameter = None,
        bias_param: nn.Parameter = None,
    ):
        """
        Internal method to compute norms. Can be called in async stream.
        
        This method is stream-safe: it doesn't access any parameter attributes
        that could trigger FSDP synchronization.
        """
        if self.use_ghost_clipping and (
            module_type in self.NORM_SAMPLERS or 
            (self.use_flash_clipping and module_type in self.FLASH_NORM_SAMPLERS)
        ):
            # Use Flash sampler if available and enabled, otherwise use standard sampler
            if self.use_flash_clipping and module_type in self.FLASH_NORM_SAMPLERS:
                # Import async version for Linear layers
                if module_type == nn.Linear:
                    from opacus.grad_sample.triton_kernels_async import compute_linear_norm_sample_flash_async
                    norm_samples = compute_linear_norm_sample_flash_async(
                        activations=activations,
                        backprops=backprops,
                        weight_requires_grad=weight_requires_grad,
                        bias_requires_grad=bias_requires_grad,
                        weight_param=weight_param,
                        bias_param=bias_param,
                        algorithm="input_length",
                        use_flash_clipping=self.use_flash_clipping,
                    )
                else:
                    # Other flash samplers (not Linear)
                    norm_sampler_fn = self.FLASH_NORM_SAMPLERS[module_type]
                    norm_samples = norm_sampler_fn(module, activations, backprops)
            else:
                # Standard norm sampler
                norm_sampler_fn = self.NORM_SAMPLERS[module_type]
                norm_samples = norm_sampler_fn(module, activations, backprops)

            # Assign norms to module.norm_sample
            # IMPORTANT: Must filter by requires_grad and iterate in same order as original
            for idx, (param, ns) in enumerate(
                (item for item in norm_samples.items() if item[0].requires_grad)
            ):
                if idx < len(module.norm_sample):
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
            # Fall back to grad sampler
            if not self.force_functorch and module_type in self.GRAD_SAMPLERS:
                grad_sampler_fn = self.GRAD_SAMPLERS[module_type]
            else:
                grad_sampler_fn = ft_compute_per_sample_gradient

            grad_samples = grad_sampler_fn(module, activations, backprops)

            for idx, (_, gs) in enumerate((item for item in grad_samples.items())):
                module.norm_sample[idx] = gs.reshape(len(gs), -1).norm(2, dim=-1)
            del grad_samples
    
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
            if backprops.dim() == 2:
                # [B, d] -> multiply by [B, 1]
                clipped_backprops = backprops * clipping_coef.view(-1, 1).to(device=backprops.device, dtype=backprops.dtype)
            elif backprops.dim() == 3:
                # [B, T, d] -> multiply by [B, 1, 1]
                clipped_backprops = backprops * clipping_coef.view(-1, 1, 1).to(device=backprops.device, dtype=backprops.dtype)
            else:
                raise ValueError(f"Unsupported backprops dimension: {backprops.dim()}")
            
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

