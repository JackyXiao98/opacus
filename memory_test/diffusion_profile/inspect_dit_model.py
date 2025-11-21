#!/usr/bin/env python3
"""
Inspect DiT Model Structure and Opacus Compatibility

This script loads facebook/DiT-XL-2-256 model from diffusers and analyzes its structure
to identify layers that need to be fixed for Opacus compatibility.
"""

import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))

import torch
import torch.nn as nn
from opacus.validators import ModuleValidator
from typing import List, Tuple


def print_separator(char='=', length=80):
    """Print a separator line"""
    print(char * length)


def print_module_info(name: str, module: nn.Module, indent: int = 0):
    """
    Print detailed information about a module.
    
    Args:
        name: Module name
        module: PyTorch module
        indent: Indentation level
    """
    prefix = "  " * indent
    module_type = type(module).__name__
    
    # Get module parameters info
    params_info = []
    
    if hasattr(module, 'embed_dim'):
        params_info.append(f"embed_dim={module.embed_dim}")
    if hasattr(module, 'num_heads'):
        params_info.append(f"num_heads={module.num_heads}")
    if hasattr(module, 'hidden_size'):
        params_info.append(f"hidden_size={module.hidden_size}")
    if hasattr(module, 'num_attention_heads'):
        params_info.append(f"num_heads={module.num_attention_heads}")
    if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
        params_info.append(f"in={module.in_features}, out={module.out_features}")
    if hasattr(module, 'num_channels'):
        params_info.append(f"channels={module.num_channels}")
    
    params_str = f" ({', '.join(params_info)})" if params_info else ""
    
    print(f"{prefix}{name}: {module_type}{params_str}")


def check_module_compatibility(module: nn.Module) -> Tuple[bool, List[str]]:
    """
    Check if a module is compatible with Opacus.
    
    Args:
        module: PyTorch module to check
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = ModuleValidator.validate(module, strict=False)
    is_valid = len(errors) == 0
    error_messages = [str(e) for e in errors]
    return is_valid, error_messages


def inspect_model_structure(model: nn.Module, max_depth: int = 10):
    """
    Recursively inspect model structure.
    
    Args:
        model: PyTorch model
        max_depth: Maximum depth to recurse
    """
    print_separator()
    print("MODEL STRUCTURE")
    print_separator()
    
    def recurse(module, name, depth):
        if depth > max_depth:
            return
        
        print_module_info(name if name else "Root", module, depth)
        
        # Check if this specific module type has known issues
        if isinstance(module, nn.MultiheadAttention):
            print("  " * (depth + 1) + "⚠️  INCOMPATIBLE: Needs replacement with DPMultiheadAttention")
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            print("  " * (depth + 1) + "⚠️  INCOMPATIBLE: BatchNorm needs replacement")
        elif isinstance(module, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            print("  " * (depth + 1) + "⚠️  INCOMPATIBLE: InstanceNorm needs replacement")
        
        # Recurse into children
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            recurse(child_module, full_name, depth + 1)
    
    recurse(model, "", 0)


def inspect_dit_model():
    """Main function to inspect DiT model"""
    print("\n" + "=" * 80)
    print("DiT Model Inspection for Opacus Compatibility")
    print("=" * 80 + "\n")
    
    try:
        from diffusers import DiTPipeline
        
        print("Loading facebook/DiT-XL-2-256 from diffusers...")
        pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float32)
        
        # Extract the transformer model from the pipeline
        model = pipe.transformer
        
        print(f"DiT-XL-2-256 Architecture:")
        print(f"  Hidden size: {model.config.hidden_size if hasattr(model.config, 'hidden_size') else 'N/A'}")
        print(f"  Num layers: {model.config.num_layers if hasattr(model.config, 'num_layers') else 'N/A'}")
        print(f"  Num attention heads: {model.config.num_attention_heads if hasattr(model.config, 'num_attention_heads') else 'N/A'}")
        print(f"  Sample size: {model.config.sample_size if hasattr(model.config, 'sample_size') else 'N/A'}")
        print(f"  Patch size: {model.config.patch_size if hasattr(model.config, 'patch_size') else 'N/A'}")
        print(f"  In channels: {model.config.in_channels if hasattr(model.config, 'in_channels') else 'N/A'}")
        
        model.eval()  # Set to eval mode for inspection
        
        print("\n")
        
        # Inspect structure
        inspect_model_structure(model, max_depth=4)
        
        print("\n")
        print_separator()
        print("OPACUS COMPATIBILITY CHECK")
        print_separator()
        
        # Set to training mode for validation
        model.train()
        
        # Check overall compatibility
        is_valid, errors = check_module_compatibility(model)
        
        if is_valid:
            print("✅ Model is compatible with Opacus!")
        else:
            print(f"❌ Model has {len(errors)} compatibility issue(s):\n")
            for i, error in enumerate(errors, 1):
                print(f"{i}. {error}")
        
        print("\n")
        print_separator()
        print("TRAINABLE PARAMETERS")
        print_separator()
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params:,}")
        
        # Count parameters by layer type
        layer_types = {}
        for name, module in model.named_modules():
            module_type = type(module).__name__
            num_params = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
            if num_params > 0:
                if module_type not in layer_types:
                    layer_types[module_type] = {'count': 0, 'params': 0}
                layer_types[module_type]['count'] += 1
                layer_types[module_type]['params'] += num_params
        
        print("\nParameters by layer type:")
        for layer_type, info in sorted(layer_types.items(), key=lambda x: x[1]['params'], reverse=True):
            print(f"  {layer_type}: {info['count']} instances, {info['params']:,} params")
        
        print("\n")
        print_separator()
        print("FIXING RECOMMENDATION")
        print_separator()
        
        if not is_valid:
            print("To make this model compatible with Opacus:")
            print("1. Use ModuleValidator.fix(model, use_flash_attention=True)")
            print("2. This will replace incompatible layers with DP-compatible versions")
            print("3. Verify with ModuleValidator.is_valid(fixed_model)")
            
            print("\nExample code:")
            print("""
from diffusers import DiTPipeline
from opacus.validators import ModuleValidator

pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float32)
model = pipe.transformer
fixed_model = ModuleValidator.fix(model, use_flash_attention=True)
assert ModuleValidator.is_valid(fixed_model)
            """)
        
        return model
        
    except ImportError as e:
        print(f"❌ Error: diffusers library not found")
        print(f"   Install with: pip install diffusers")
        print(f"   Details: {e}")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    inspect_dit_model()

