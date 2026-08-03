"""
Unified attention injection for any HuggingFace model.
Uses architecture helpers and weight copying to swap attention modules.
"""

from typing import Optional

import torch.nn as nn

from .adapters import get_adapter
from .architecture import detect_architecture, get_attention_modules
from .model_utils import replace_module
from .registry import ATTENTION_REGISTRY


def inject_attention(
    model: nn.Module,
    attention_type: str,
    rank: Optional[int] = None,
    copy_weights: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Replace all attention modules in the model with the specified variant.
    Automatically copies pretrained weights using the appropriate adapter.
    """
    attn_class = ATTENTION_REGISTRY.resolve(attention_type)
    config = getattr(model, "config", None)
    arch = detect_architecture(model)
    adapter = get_adapter(arch)

    replaced = 0
    for name, old_attn in get_attention_modules(model):
        # Create new attention instance
        new_attn = None
        signatures_to_try = []
        if config is not None:
            signatures_to_try.append(
                lambda: attn_class(
                    config, rank=rank, pretrained_attn=old_attn, **kwargs
                )
            )
            signatures_to_try.append(lambda: attn_class(config, rank=rank, **kwargs))
            signatures_to_try.append(lambda: attn_class(config, **kwargs))
            signatures_to_try.append(lambda: attn_class(config))
        else:
            signatures_to_try.append(lambda: attn_class(rank=rank, **kwargs))
            signatures_to_try.append(lambda: attn_class(**kwargs))
            signatures_to_try.append(lambda: attn_class())

        for fn in signatures_to_try:
            try:
                new_attn = fn()
                break
            except TypeError:
                continue

        if new_attn is None:
            if config is not None:
                new_attn = attn_class(config)
            else:
                new_attn = attn_class()

        # Copy pretrained weights
        if copy_weights:
            for proj in ["q", "k", "v", "o"]:
                try:
                    src_weight, src_bias = adapter.get_weight_bias(old_attn, proj)
                except ValueError:
                    continue  # skip if projection not found
                target_proj = getattr(new_attn, f"{proj}_proj", None)
                if target_proj is None:
                    target_proj = getattr(new_attn, proj, None)
                if target_proj is not None:
                    target_proj.weight.data.copy_(src_weight)
                    if src_bias is not None and target_proj.bias is not None:
                        target_proj.bias.data.copy_(src_bias)
                    elif src_bias is None and target_proj.bias is not None:
                        nn.init.zeros_(target_proj.bias)

        replace_module(model, name, new_attn)
        replaced += 1

    print(f"✅ Replaced {replaced} attention modules with '{attention_type}'.")
    return model


def available_attention_types() -> list:
    return ATTENTION_REGISTRY.keys()
