"""
Unified attention injection for any HuggingFace model.
Uses architecture helpers and weight copying to swap attention modules.
"""

import torch.nn as nn
from typing import Optional
from .registry import ATTENTION_REGISTRY
from .architecture import get_attention_modules
from .model_utils import copy_attention_weights, replace_module


def inject_attention(
    model: nn.Module,
    attention_type: str,
    rank: Optional[int] = None,
    copy_weights: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Replace all attention modules in the model with the specified variant.
    """
    attn_class = ATTENTION_REGISTRY.resolve(attention_type)
    config = getattr(model, 'config', None)
    replaced = 0

    for name, old_attn in get_attention_modules(model):
        # Create new attention instance
        try:
            if config is not None:
                new_attn = attn_class(config, rank=rank, pretrained_attn=old_attn, **kwargs)
            else:
                new_attn = attn_class(rank=rank, **kwargs)
        except TypeError:
            new_attn = attn_class(rank=rank, **kwargs)

        # Copy weights if requested
        if copy_weights:
            try:
                copy_attention_weights(old_attn, new_attn)
            except Exception as e:
                print(f"  ⚠️ Could not copy weights for '{name}': {e}")

        replace_module(model, name, new_attn)
        replaced += 1

    print(f"✅ Replaced {replaced} attention modules with '{attention_type}'.")
    return model


def available_attention_types() -> list:
    """Return all registered attention variant keys."""
    return ATTENTION_REGISTRY.keys()