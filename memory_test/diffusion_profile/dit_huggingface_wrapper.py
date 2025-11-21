#!/usr/bin/env python3
"""
DiT Model Wrapper for DP-SGD Experiments

This module provides a DiT model wrapper that uses DP-compatible attention layers
for Opacus DP-SGD framework. It uses facebook/DiT-XL-2-256 from diffusers library
as the reference architecture.
"""

import torch
import torch.nn as nn
from typing import Optional
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))


class DiTHuggingFaceWrapper(nn.Module):
    """
    DP-compatible DiT model wrapper.
    
    This wrapper uses a custom DiT implementation with DP-compatible attention layers
    based on the facebook/DiT-XL-2-256 architecture from diffusers.
    
    Features:
    1. Uses DPMultiheadAttentionWithFlashAttention for efficient per-sample gradients
    2. Handles forward pass with diffusion inputs (images, timesteps, labels)
    3. Returns dict with loss when target_noise is provided (vanilla training)
    4. Returns tensor when target_noise is None (DP-SGD mode)
    """
    
    def __init__(
        self,
        model_name: str = "facebook/DiT-XL-2-256",
        img_size: int = 256,
        patch_size: int = 2,
        in_channels: int = 4,
        num_classes: int = 1000,
        pretrained: bool = False,
        use_flash_attention: bool = True,
    ):
        """
        Args:
            model_name: Diffusers model identifier (e.g., facebook/DiT-XL-2-256)
            img_size: Input image size (height and width)
            patch_size: Patch size for tokenization
            in_channels: Number of input channels (4 for latent space with pretrained model)
            num_classes: Number of label classes
            pretrained: Whether to load pretrained weights (from diffusers)
                       NOTE: If pretrained=True, in_channels must be 4 (latent space inputs)
                       If pretrained=False, uses custom DP-compatible implementation that works
                       with any number of channels
            use_flash_attention: Whether to use Flash Attention for memory efficiency (default: True)
        """
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.model_name = model_name
        self.use_flash_attention = use_flash_attention
        self.using_diffusers_model = False
        
        if pretrained:
            # Try to load and fix diffusers model
            print(f"Attempting to load pretrained model: {model_name}")
            print(f"⚠️  NOTE: Pretrained DiT model requires 4-channel latent space inputs")
            if in_channels != 4:
                print(f"⚠️  WARNING: in_channels={in_channels} but pretrained model expects 4 channels!")
                print(f"⚠️  Consider setting pretrained=False to use custom DP-compatible implementation")
            self.dit_model = self._load_and_fix_diffusers_model()
        else:
            # Use custom DP-compatible implementation
            print(f"Creating DP-compatible DiT model (config reference: {model_name})")
            self.dit_model = self._create_fallback_model()
    
    def _load_and_fix_diffusers_model(self):
        """
        Try to load diffusers DiT model and fix it for Opacus compatibility.
        Falls back to custom implementation if loading fails.
        """
        try:
            from diffusers import DiTPipeline
            from opacus.validators import ModuleValidator
            
            print("Loading DiT model from diffusers...")
            
            # Try to load pretrained model
            try:
                # Try to enable Flash Attention 2 in diffusers if available
                load_kwargs = {"torch_dtype": torch.float32}
                
                if self.use_flash_attention:
                    try:
                        # Check if flash_attn is available
                        import importlib
                        flash_attn_spec = importlib.util.find_spec("flash_attn")
                        if flash_attn_spec is not None:
                            print("  Enabling Flash Attention 2 from diffusers...")
                            load_kwargs["use_flash_attention_2"] = True
                        else:
                            print("  flash_attn not found, will use Opacus Flash Attention instead")
                    except Exception:
                        print("  Could not enable Flash Attention 2, will use Opacus version")
                
                pipe = DiTPipeline.from_pretrained(self.model_name, **load_kwargs)
                # Extract the transformer model from the pipeline
                model = pipe.transformer
                print(f"✓ Successfully loaded pretrained DiT from {self.model_name}")
                        
            except Exception as e:
                print(f"⚠ Could not load pretrained weights: {e}")
                print("  Falling back to custom implementation...")
                return self._create_fallback_model()
            
            # Freeze parameters that are shared (tied) to avoid Ghost Clipping issues
            print("Checking for parameter tying (shared parameters)...")
            self._freeze_tied_parameters(model)
            
            # Check compatibility and apply fixes
            print("Checking Opacus compatibility...")
            model.train()  # Must be in training mode for validation
            errors = ModuleValidator.validate(model, strict=False)
            
            # Always apply Flash Attention if requested, even if model is compatible
            if self.use_flash_attention or len(errors) > 0:
                if len(errors) > 0:
                    print(f"  Found {len(errors)} incompatible layer(s)")
                    print(f"  Applying automatic fixes with Flash Attention...")
                else:
                    print(f"  Model is compatible, but applying Flash Attention optimization...")
                
                # Fix the model with Flash Attention
                fixed_model = ModuleValidator.fix(model, use_flash_attention=self.use_flash_attention)
                
                # Validate the fixed model
                if ModuleValidator.is_valid(fixed_model):
                    print("✓ Model successfully fixed and validated with Flash Attention!")
                    self.using_diffusers_model = True
                    return fixed_model
                else:
                    print("✗ Model validation failed after fixing")
                    raise Exception("Fixed model still has compatibility issues")
            else:
                print("✓ Model is already compatible!")
                self.using_diffusers_model = True
                return model
                
        except ImportError as e:
            print(f"✗ diffusers library not found: {e}")
            print("  Install with: pip install diffusers")
            print("  Falling back to custom implementation...")
            return self._create_fallback_model()
        except Exception as e:
            print(f"✗ Failed to load/fix diffusers model: {e}")
            print("  Falling back to custom implementation...")
            return self._create_fallback_model()
    
    def _freeze_tied_parameters(self, model):
        """
        Detect and freeze tied (shared) parameters to avoid Ghost Clipping issues.
        Ghost Clipping doesn't support parameter tying.
        Uses data_ptr() which is more reliable than id() for detecting shared tensors.
        """
        # Track parameter data pointers (more reliable than object id)
        data_ptr_to_names = {}
        param_list = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                data_ptr = param.data_ptr()
                if data_ptr not in data_ptr_to_names:
                    data_ptr_to_names[data_ptr] = []
                data_ptr_to_names[data_ptr].append(name)
                param_list.append((name, param))
        
        # Find tied parameters (same underlying data)
        tied_params = {ptr: names for ptr, names in data_ptr_to_names.items() if len(names) > 1}
        
        frozen_count = 0
        if tied_params:
            print(f"  Found {len(tied_params)} tied parameter group(s)")
            for data_ptr, names in tied_params.items():
                print(f"    Tied parameters: {names[:3]}{'...' if len(names) > 3 else ''}")
                # Freeze all parameters with this data pointer
                for n, p in param_list:
                    if p.data_ptr() == data_ptr and p.requires_grad:
                        p.requires_grad = False
                        frozen_count += 1
                        print(f"      -> Freezing {n}")
            print(f"  Total frozen parameters: {frozen_count}")
        else:
            print("  No tied parameters detected by data_ptr()")
            
            # Additional heuristic: freeze known problematic layers in DiT models
            print("  Applying heuristic: freezing embedding layers that might be shared...")
            for name, param in param_list:
                # Freeze timestep embedders and class embedders which are often shared
                if any(keyword in name for keyword in [
                    'timestep_embedder', 
                    'class_embedder',
                    'time_embed',
                    'label_emb'
                ]):
                    if param.requires_grad:
                        param.requires_grad = False
                        frozen_count += 1
                        print(f"      -> Freezing {name}")
            
            if frozen_count > 0:
                print(f"  Total frozen parameters (heuristic): {frozen_count}")
            else:
                print("  No embedding layers found to freeze")
    
    def _create_fallback_model(self):
        """Create a fallback transformer-based model with DiT-XL-2-256 architecture"""
        from memory_test.diffusion_dit_bookkeeping.diffusion_dit_model import DiTModelWithFlashAttention
        
        # DiT-XL-2-256 architecture parameters
        # XL = Extra Large: 1152 hidden dim, 28 layers, 16 heads
        return DiTModelWithFlashAttention(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            hidden_dim=1152,  # DiT-XL default
            num_layers=28,    # DiT-XL depth
            num_heads=16,
            num_classes=self.num_classes,
        )
    
    def forward(
        self,
        images: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor,
        target_noise: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for diffusion training.
        
        Args:
            images: Input images [B, C, H, W]
            timesteps: Diffusion timesteps [B]
            labels: Class labels [B]
            target_noise: Target noise for loss computation [B, C, H, W] (optional)
        
        Returns:
            When target_noise is provided: dict with 'loss' and 'logits'
            When target_noise is None: tensor [B, C, H, W] (DP-SGD mode)
        """
        if self.using_diffusers_model:
            # Diffusers model expects (hidden_states, timestep, class_labels)
            # and returns .sample tensor
            output = self.dit_model(
                hidden_states=images,
                timestep=timesteps,
                class_labels=labels,
                return_dict=True
            )
            predicted_noise = output.sample
            
            # DiT models may output double channels (mean + variance or for classifier-free guidance)
            # Extract only the first in_channels
            if predicted_noise.shape[1] > self.in_channels:
                predicted_noise = predicted_noise[:, :self.in_channels, :, :]
            
            # Handle target_noise for loss computation
            if target_noise is not None:
                # Compute MSE loss
                loss = torch.nn.functional.mse_loss(predicted_noise, target_noise)
                return {"loss": loss, "logits": predicted_noise}
            else:
                # DP-SGD mode: return tensor directly
                return predicted_noise
        else:
            # Use the DP-compatible fallback model
            return self.dit_model(images, timesteps, labels, target_noise)


def create_dit_huggingface_model(
    img_size: int = 256,
    patch_size: int = 2,
    in_channels: int = 4,
    num_classes: int = 1000,
    pretrained: bool = False,
    use_flash_attention: bool = True,
    device: str = "cuda",
) -> DiTHuggingFaceWrapper:
    """
    Factory function to create a DiT model based on facebook/DiT-XL-2-256.
    
    Args:
        img_size: Input image size (256 for DiT-XL-2-256)
        patch_size: Patch size (2 for DiT-XL-2-256)
        in_channels: Number of input channels (4 for latent space)
        num_classes: Number of classes
        pretrained: Whether to load pretrained weights
        use_flash_attention: Enable Flash Attention for memory efficiency
        device: Device to move model to
    
    Returns:
        DiTHuggingFaceWrapper instance
    """
    model = DiTHuggingFaceWrapper(
        model_name="facebook/DiT-XL-2-256",
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=pretrained,
        use_flash_attention=use_flash_attention,
    )
    
    return model.to(device)

