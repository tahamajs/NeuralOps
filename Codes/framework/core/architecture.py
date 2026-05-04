# framework/core/architecture.py
"""
Architecture detection and attention replacement utilities.
"""

import torch.nn as nn
from typing import List, Tuple, Any, Optional, Union
from transformers import PreTrainedModel

# ----------------------------------------------------------------------
# Architecture detection
# ----------------------------------------------------------------------

def detect_architecture(model: nn.Module) -> str:
    """
    Detect Transformer architecture family.
    
    Returns:
        One of: 'bert', 'roberta', 'distilbert', 'gpt2', 'gpt_neo', 
        'gptj', 'llama', 'mistral', 't5', 'gemma', 'falcon', 'qwen', 'unknown'
    """
    model_cls = model.__class__.__name__.lower()
    config = getattr(model, 'config', None)
    model_type = getattr(config, 'model_type', '')
    
    # 1) BERT family
    if 'bert' in model_type or 'bert' in model_cls:
        return 'bert'
    if 'roberta' in model_type or 'roberta' in model_cls:
        return 'roberta'
    if 'distilbert' in model_type or 'distilbert' in model_cls:
        return 'distilbert'
    
    # 2) GPT family
    if 'gpt2' in model_type or 'gpt2' in model_cls:
        return 'gpt2'
    if 'gpt_neo' in model_type or 'gpt_neo' in model_cls:
        return 'gpt_neo'
    if 'gptj' in model_type or 'gpt_j' in model_type:
        return 'gptj'
    
    # 3) LLaMA family
    if 'llama' in model_type:
        return 'llama'
    if 'mistral' in model_type:
        return 'mistral'
    
    # 4) T5 / encoder‑decoder
    if 't5' in model_type:
        return 't5'
    
    # 5) Others
    if 'gemma' in model_type:
        return 'gemma'
    if 'falcon' in model_type:
        return 'falcon'
    if 'qwen' in model_type:
        return 'qwen'
    
    return 'unknown'

# ----------------------------------------------------------------------
# Extract transformer layers
# ----------------------------------------------------------------------

def get_transformer_layers(model: nn.Module) -> List[nn.Module]:
    """
    Return the list of transformer block modules (regardless of architecture).
    """
    arch = detect_architecture(model)
    # Standard HF naming
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT‑style
        return model.transformer.h
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'layer'):
        # Some models use 'layer'
        return model.transformer.layer
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        return model.encoder.layer
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'layer'):
        return model.decoder.layer
    if hasattr(model, 'bert') and hasattr(model.bert, 'encoder'):
        return model.bert.encoder.layer
    if hasattr(model, 'roberta') and hasattr(model.roberta, 'encoder'):
        return model.roberta.encoder.layer
    if hasattr(model, 'distilbert') and hasattr(model.distilbert, 'transformer'):
        return model.distilbert.transformer.layer
    
    return []

# ----------------------------------------------------------------------
# Find attention modules
# ----------------------------------------------------------------------

def get_attention_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Find all attention submodules in the model.
    
    Returns:
        List of (module_path, module) pairs
    """
    attention_modules = []
    for name, module in model.named_modules():
        # Typical attention class names in HF transformers
        if 'attention' in name.lower() or 'attn' in name.lower():
            # Exclude things like 'attention_mask'
            if not any(x in name for x in ['mask', 'bias', 'norm']):
                attention_modules.append((name, module))
    return attention_modules

# ----------------------------------------------------------------------
# Replace attention modules
# ----------------------------------------------------------------------

def replace_attention_module(
    model: nn.Module,
    new_attention_class: Any,
    rank: Optional[int] = None,
    **kwargs
) -> None:
    """
    Replace every attention module in the model with an instance of `new_attention_class`.
    
    Args:
        model: The model to modify (in‑place)
        new_attention_class: Attention block class (registered in ATTENTION_REGISTRY)
        rank: Optional rank (may be used by the block)
        **kwargs: Additional arguments passed to the block constructor
    """
    config = getattr(model, 'config', None)
    # Build constructor args
    if rank is not None and 'rank' not in kwargs:
        kwargs['rank'] = rank
    
    for name, module in get_attention_modules(model):
        # Get the parent module and attribute name
        *parent_parts, child_name = name.split('.')
        parent = model
        for part in parent_parts:
            parent = getattr(parent, part)
        
        # Instantiate new attention block
        try:
            if config is not None:
                new_module = new_attention_class(config, **kwargs)
            else:
                new_module = new_attention_class(**kwargs)
        except TypeError:
            # Fallback: some blocks may not accept config
            new_module = new_attention_class(**kwargs)
        
        # Replace
        setattr(parent, child_name, new_module)