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

from typing import Iterable

import torch
import torch.nn as nn
from opacus.grad_sample.grad_sample_module_fast_gradient_clipping_fsdp import (
    GradSampleModuleFastGradientClippingFSDP,
)
from opacus.utils.module_utils import has_trainable_params
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard


def has_params(module: nn.Module) -> bool:
    return len(list(module.parameters(recurse=False))) > 0


def is_transformer_block(module: nn.Module) -> bool:
    """
    Detect if a module is a Transformer block that should be wrapped as a unit.
    
    This function identifies common Transformer block types from popular architectures
    to enable coarse-grained FSDP wrapping, which reduces communication overhead.
    
    Recognized block types:
    - LlamaDecoderLayer (Llama/Llama2/Llama3 models)
    - GPT2Block (GPT-2 models)
    - BertLayer (BERT models)
    - TransformerBlock (generic Transformer blocks)
    - TransformerEncoderLayer (PyTorch native)
    - TransformerDecoderLayer (PyTorch native)
    
    Args:
        module: PyTorch module to check
        
    Returns:
        True if module is a recognized Transformer block type
    """
    module_name = type(module).__name__
    transformer_block_names = [
        'LlamaDecoderLayer',
        'Llama2DecoderLayer', 
        'Llama3DecoderLayer',
        'GPT2Block',
        'GPTNeoBlock',
        'GPTJBlock',
        'BertLayer',
        'RobertaLayer',
        'TransformerBlock',
        'TransformerEncoderLayer',
        'TransformerDecoderLayer',
        'T5Block',
        'BloomBlock',
        'OPTDecoderLayer',
        'FalconDecoderLayer',
        'DiTBlock',  # Diffusion Transformer block
    ]
    return module_name in transformer_block_names


def iterate_submodules(module: nn.Module) -> Iterable[nn.Module]:
    if has_params(module):
        yield module

    for m in module.children():
        yield from iterate_submodules(m)


def _is_inside_wrapped_block(module: nn.Module, wrapped_blocks: set) -> bool:
    """
    Check if a module is contained inside any of the wrapped Transformer blocks.
    
    Args:
        module: Module to check
        wrapped_blocks: Set of modules that have been wrapped as blocks
        
    Returns:
        True if module is a descendant of any wrapped block
    """
    for wrapped_block in wrapped_blocks:
        # Check if module is a descendant of wrapped_block
        for descendant in wrapped_block.modules():
            if descendant is module:
                return True
    return False


def FSDP2Wrapper(model: nn.Module, **kwargs) -> nn.Module:
    """
    Wrap a model with FSDP2 (Fully Sharded Data Parallel v2).
    
    This function uses a two-pass wrapping strategy for optimal performance:
    1. First pass: Wrap entire Transformer blocks (coarse-grained)
    2. Second pass: Wrap remaining individual layers not inside blocks (fine-grained)
    
    This reduces communication overhead significantly by minimizing the number of
    FSDP units (e.g., from 100+ individual layers to ~16 transformer blocks).
    
    Args:
        model: The model to wrap
        **kwargs: Additional arguments including:
            - mp_policy: Mixed precision policy
            - mesh: Device mesh for multi-node training
            - opacus_high_precision_layers: List of layer types requiring higher precision
            - reshard_after_forward: Whether to free parameters after forward (default: True)
            - use_block_wrapping: Enable block-level wrapping (default: True)
    
    Returns:
        FSDP-wrapped model
    """
    sampler_classes = set(
        list(GradSampleModuleFastGradientClippingFSDP.GRAD_SAMPLERS.keys())
        + list(GradSampleModuleFastGradientClippingFSDP.NORM_SAMPLERS.keys())
    )
    mp_policy = kwargs.get("mp_policy", MixedPrecisionPolicy())
    mesh = kwargs.get("mesh", None)
    opacus_high_precision_layers = kwargs.get("opacus_high_precision_layers", [])
    reshard_after_forward = kwargs.get("reshard_after_forward", True)
    use_block_wrapping = kwargs.get("use_block_wrapping", True)  # Enable by default
    wrap_individual_layers = kwargs.get("wrap_individual_layers", True)  # Enable by default
    
    wrapped_blocks = set()
    
    # PASS 1: Wrap Transformer blocks as coarse-grained units
    # This significantly reduces communication overhead for Transformer-based models
    if use_block_wrapping:
        block_count = 0
        # Debug: collect module types
        import os
        debug_mode = os.environ.get('OPACUS_PROFILE_FSDP', '0') == '1'
        module_types_seen = set()
        llama_decoder_count = 0
        
        for module in model.modules():
            module_type_name = type(module).__name__
            if debug_mode:
                module_types_seen.add(module_type_name)
                if 'LlamaDecoderLayer' in module_type_name:
                    llama_decoder_count += 1
                    has_params_val = has_trainable_params(module)  # recurse=False
                    has_params_recursive = any(p.requires_grad for p in module.parameters(recurse=True))
                    is_block_val = is_transformer_block(module)
                    if llama_decoder_count <= 2:  # Only log first 2
                        print(f"[FSDP Wrapper Debug] Found {module_type_name}: "
                              f"has_trainable_params(recurse=False)={has_params_val}, "
                              f"has_trainable_params(recurse=True)={has_params_recursive}, "
                              f"is_transformer_block={is_block_val}")
            
            if is_transformer_block(module):
                # For Transformer blocks, check if they have parameters recursively
                # since blocks like LlamaDecoderLayer don't have direct parameters,
                # all parameters are in submodules (self_attn, mlp, etc.)
                has_params_recursive = any(p.requires_grad for p in module.parameters(recurse=True))
                if has_params_recursive:
                    fully_shard(
                        module,
                        mesh=mesh,
                        mp_policy=mp_policy,
                        reshard_after_forward=reshard_after_forward,
                    )
                    wrapped_blocks.add(module)
                    block_count += 1
        
        # Log wrapping statistics for debugging
        if debug_mode:
            try:
                import torch.distributed as dist
                if dist.is_initialized() and dist.get_rank() == 0:
                    print(f"[FSDP Wrapper] Found {llama_decoder_count} LlamaDecoderLayer modules")
                    print(f"[FSDP Wrapper] Wrapped {block_count} Transformer blocks as coarse-grained units")
            except:
                print(f"[FSDP Wrapper] Found {llama_decoder_count} LlamaDecoderLayer modules")
                print(f"[FSDP Wrapper] Wrapped {block_count} Transformer blocks as coarse-grained units")
    
    # PASS 2: Wrap remaining individual layers (only those not inside wrapped blocks)
    # This handles leaf layers outside of Transformer blocks (e.g., embedding, final layer)
    layer_count = 0
    if wrap_individual_layers:
        for module in iterate_submodules(model):
            # Skip if module is a wrapped block or is inside a wrapped block
            if module in wrapped_blocks or _is_inside_wrapped_block(module, wrapped_blocks):
                continue
                
            if (type(module) in sampler_classes) or (not has_trainable_params(module)):
                if len(opacus_high_precision_layers) > 0 and isinstance(
                    module, opacus_high_precision_layers
                ):
                    # For certain layers, higher precision is needed to stabilize the training of DP-SGD.
                    fully_shard(
                        module,
                        mesh=mesh,
                        mp_policy=MixedPrecisionPolicy(
                            param_dtype=torch.get_default_dtype()
                        ),
                        reshard_after_forward=reshard_after_forward,
                    )
                else:
                    fully_shard(
                        module, 
                        mesh=mesh, 
                        mp_policy=mp_policy,
                        reshard_after_forward=reshard_after_forward,
                    )
                layer_count += 1
    
    # Log wrapping statistics
    import os
    if os.environ.get('OPACUS_PROFILE_FSDP', '0') == '1':
        try:
            import torch.distributed as dist
            if dist.is_initialized() and dist.get_rank() == 0:
                print(f"[FSDP Wrapper] Wrapped {layer_count} individual layers outside blocks")
        except:
            print(f"[FSDP Wrapper] Wrapped {layer_count} individual layers outside blocks")
    
    # Finally, wrap the root model
    model = fully_shard(
        model, 
        mesh=mesh, 
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
    )
    return model
