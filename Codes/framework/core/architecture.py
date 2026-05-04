# framework/core/architecture.py
import torch.nn as nn
from typing import List, Tuple, Optional

def detect_architecture(model: nn.Module) -> str:
    config = getattr(model, 'config', None)
    model_type = getattr(config, 'model_type', '')
    name = model.__class__.__name__.lower()
    if 'bert' in model_type or 'bert' in name:
        return 'bert'
    if 'roberta' in model_type or 'roberta' in name:
        return 'roberta'
    if 'distilbert' in model_type or 'distilbert' in name:
        return 'distilbert'
    if 'gpt2' in model_type or 'gpt2' in name:
        return 'gpt2'
    if 'llama' in model_type:
        return 'llama'
    if 'mistral' in model_type:
        return 'mistral'
    return 'unknown'

def get_transformer_layers(model: nn.Module) -> List[nn.Module]:
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'layer'):
        return model.transformer.layer
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        return model.encoder.layer
    if hasattr(model, 'bert') and hasattr(model.bert, 'encoder'):
        return model.bert.encoder.layer
    if hasattr(model, 'roberta') and hasattr(model.roberta, 'encoder'):
        return model.roberta.encoder.layer
    if hasattr(model, 'distilbert') and hasattr(model.distilbert, 'transformer'):
        return model.distilbert.transformer.layer
    return []

def get_attention_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    modules = []
    for name, module in model.named_modules():
        if 'attention' in name.lower() or 'attn' in name.lower():
            if not any(x in name for x in ['mask', 'bias', 'norm']):
                modules.append((name, module))
    return modules

def replace_attention_module(model: nn.Module, new_attention_class, rank: Optional[int] = None, **kwargs):
    config = getattr(model, 'config', None)
    if rank is not None and 'rank' not in kwargs:
        kwargs['rank'] = rank
    for name, module in get_attention_modules(model):
        *parent_parts, child_name = name.split('.')
        parent = model
        for part in parent_parts:
            parent = getattr(parent, part)
        try:
            if config:
                new_module = new_attention_class(config, **kwargs)
            else:
                new_module = new_attention_class(**kwargs)
        except TypeError:
            new_module = new_attention_class(**kwargs)
        setattr(parent, child_name, new_module)