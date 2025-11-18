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
import time
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
        enable_profiling = os.environ.get('OPACUS_PROFILE_FSDP', '0') == '1'
        sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
        
        if enable_profiling:
            sync()
            t_start = time.time()
        
        # Stack per-parameter norms from all modules
        stacked_norms = torch.stack(
            [
                per_param_norm
                for module in self.iterate_submodules(self._module)
                for per_param_norm in module.norm_sample
            ],
            dim=0,
        )
        
        if enable_profiling:
            sync()
            t_stack = time.time()
        
        # Compute local contribution: sum of squared norms
        # norm^2 = sum(param_i_norm^2) for parameters on this rank
        norm_sample_squared = (stacked_norms ** 2).sum(dim=0)

        if enable_profiling:
            sync()
            t_local = time.time()
            # Log pre-allreduce values for correctness debugging
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            print(f"[FSDP Profile] Rank {rank} - Pre-allreduce squared norms shape: {norm_sample_squared.shape}, "
                  f"mean: {norm_sample_squared.mean().item():.6f}, "
                  f"max: {norm_sample_squared.max().item():.6f}")

        # All-reduce the squared norms across ranks to get global per-sample gradient norms
        # This is critical for FSDP: we need sqrt(sum_all_ranks(norm_i^2)), not sum(sqrt(norm_i^2))
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(norm_sample_squared, op=torch.distributed.ReduceOp.SUM)

        if enable_profiling:
            sync()
            t_allreduce = time.time()
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            print(f"[FSDP Profile] Rank {rank} - Post-allreduce squared norms shape: {norm_sample_squared.shape}, "
                  f"mean: {norm_sample_squared.mean().item():.6f}, "
                  f"max: {norm_sample_squared.max().item():.6f}")

        # Take square root to get final per-sample gradient norms
        norm_sample = torch.sqrt(norm_sample_squared + 1e-12)  # Add epsilon for numerical stability

        if enable_profiling:
            sync()
            t_end = time.time()
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            print(f"[FSDP Profile] Rank {rank} get_norm_sample timing breakdown:")
            print(f"  - Stack norms:   {(t_stack - t_start)*1000:.2f} ms")
            print(f"  - Local compute: {(t_local - t_stack)*1000:.2f} ms")
            print(f"  - All-reduce:    {(t_allreduce - t_local)*1000:.2f} ms")
            print(f"  - Final compute: {(t_end - t_allreduce)*1000:.2f} ms")
            print(f"  - TOTAL:         {(t_end - t_start)*1000:.2f} ms")

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

        if not hasattr(module, "norm_sample"):
            # currently, we don't support freezing and unfreezing params in between training. Making this a dictionary and mapping with param names might fix this.
            module.norm_sample = []
            for _, param in trainable_parameters(module):
                module.norm_sample.append(
                    torch.zeros(
                        torch.Size([module.max_batch_len, 1]),
                        device=param.device,
                        dtype=param.dtype,
                    )
                )

        module_type = self._get_module_type(module)
        module._forward_counter -= 1
        if self.use_ghost_clipping and (
            module_type in self.NORM_SAMPLERS or 
            (self.use_flash_clipping and module_type in self.FLASH_NORM_SAMPLERS)
        ):
            # Use Flash sampler if available and enabled, otherwise use standard sampler
            if self.use_flash_clipping and module_type in self.FLASH_NORM_SAMPLERS:
                norm_sampler_fn = self.FLASH_NORM_SAMPLERS[module_type]
            else:
                norm_sampler_fn = self.NORM_SAMPLERS[module_type]
            
            # Profiling: Track per-layer norm computation time
            enable_profiling = os.environ.get('OPACUS_PROFILE_FSDP', '0') == '1'
            if enable_profiling:
                sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
                sync()
                t_start = time.time()
            
            norm_samples = norm_sampler_fn(module, activations, backprops)
            
            if enable_profiling:
                sync()
                t_end = time.time()
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                act_shape = activations[0].shape if activations else "N/A"
                bp_shape = backprops.shape
                print(f"[FSDP Profile] Rank {rank} Layer {module_type.__name__} norm computation: "
                      f"{(t_end - t_start)*1000:.2f} ms "
                      f"(act: {act_shape}, bp: {bp_shape})")

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
