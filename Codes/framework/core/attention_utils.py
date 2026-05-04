# framework/core/attention_utils.py
import torch.nn as nn
from typing import Optional
from .registry import ATTENTION_REGISTRY
from .architecture import get_attention_modules

def replace_attention_in_model(
    model: nn.Module,
    attention_type: str,
    rank: Optional[int] = None,
    copy_weights: bool = True
) -> nn.Module:
    attn_class = ATTENTION_REGISTRY.get(attention_type)
    if attn_class is None:
        raise ValueError(f"Unknown attention type: {attention_type}")
    config = getattr(model, 'config', None)
    replaced = 0
    for name, module in get_attention_modules(model):
        *parent_parts, child_name = name.split('.')
        parent = model
        for part in parent_parts:
            parent = getattr(parent, part)
        try:
            new_attn = attn_class(config, rank=rank) if config else attn_class(rank=rank)
        except TypeError:
            new_attn = attn_class(rank=rank)
        setattr(parent, child_name, new_attn)
        replaced += 1
    print(f"Replaced {replaced} attention modules with '{attention_type}'")
    return model

def get_attention_module_names(model: nn.Module) -> list:
    return [name for name, _ in get_attention_modules(model)]

def count_attention_modules(model: nn.Module) -> int:
    return len(get_attention_modules(model))