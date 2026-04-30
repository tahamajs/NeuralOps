# framework/core/attention_utils.py
"""
Utility functions for replacing attention mechanisms in models.
"""

import torch.nn as nn
from typing import Optional
from .registry import ATTENTION_REGISTRY


def replace_attention_in_model(
    model: nn.Module,
    attention_type: str,
    rank: Optional[int] = None,
    copy_weights: bool = True
) -> nn.Module:
    """
    Replace attention modules in a model with a different attention type.
    
    This function scans the model for attention modules and replaces them
    with the specified attention type from the registry.
    
    Args:
        model: The model to modify
        attention_type: Type of attention to use (must be registered)
        rank: Optional compression rank (for low-rank attentions)
        copy_weights: Whether to copy weights from original attention modules
        
    Returns:
        Modified model
        
    Example:
        >>> from framework.core import replace_attention_in_model
        >>> model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
        >>> model = replace_attention_in_model(model, "svd", rank=64)
    
    Raises:
        ValueError: If attention_type is not registered
    """
    attention_class = ATTENTION_REGISTRY.get(attention_type)
    if attention_class is None:
        available = list(ATTENTION_REGISTRY.keys())
        raise ValueError(
            f"Unknown attention type: '{attention_type}'. "
            f"Available types: {available}"
        )
    
    replaced_count = 0
    
    for module_name, module in model.named_modules():
        # Look for attention modules with standard patterns
        if _is_attention_module(module_name, module):
            try:
                # Find parent and child names
                if '.' in module_name:
                    parent_name, child_name = module_name.rsplit('.', 1)
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                    child_name = module_name
                
                # Create new attention instance
                config = getattr(model, 'config', None)
                if config is None:
                    # Try to get config from module
                    config = getattr(module, 'config', None)
                
                if config is None:
                    raise ValueError(f"Cannot find config for {module_name}")
                
                # Create new attention
                new_attn = attention_class(config, rank=rank)
                
                # Copy weights if requested
                if copy_weights:
                    _copy_attention_weights(module, new_attn)
                
                # Replace the module
                setattr(parent, child_name, new_attn)
                replaced_count += 1
                
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to replace attention in {module_name}: {e}")
    
    print(f"✅ Replaced {replaced_count} attention modules with '{attention_type}'")
    return model


def _is_attention_module(module_name: str, module: nn.Module) -> bool:
    """Check if a module is an attention module."""
    # Check by name patterns
    attention_patterns = ['attention', 'attn', 'self_attn', 'multihead']
    if any(pattern in module_name.lower() for pattern in attention_patterns):
        return True
    
    # Check by class name
    class_name = module.__class__.__name__.lower()
    if any(pattern in class_name for pattern in ['attention', 'attn']):
        return True
    
    # Check for standard attention attributes
    if hasattr(module, 'query') and hasattr(module, 'key') and hasattr(module, 'value'):
        return True
    
    return False


def _copy_attention_weights(source: nn.Module, target: nn.Module):
    """
    Copy weights from source attention to target attention.
    
    Handles different naming conventions between attention implementations.
    """
    source_weights = {}
    target_weights = {}
    
    # Collect source weights
    for name, param in source.named_parameters():
        if 'weight' in name or 'bias' in name:
            source_weights[name] = param
    
    # Try to copy to target
    for name, param in target.named_parameters():
        if name in source_weights:
            param.data = source_weights[name].data.clone()
        elif 'q_proj' in name and 'query' in source_weights:
            param.data = source_weights['query.weight'].data.clone()
        elif 'k_proj' in name and 'key' in source_weights:
            param.data = source_weights['key.weight'].data.clone()
        elif 'v_proj' in name and 'value' in source_weights:
            param.data = source_weights['value.weight'].data.clone()
        elif 'out_proj' in name and 'out_proj' in source_weights:
            param.data = source_weights['out_proj.weight'].data.clone()


def get_attention_module_names(model: nn.Module) -> list:
    """
    Get names of all attention modules in the model.
    
    Args:
        model: The model to inspect
        
    Returns:
        List of attention module names
    """
    attention_names = []
    for name, module in model.named_modules():
        if _is_attention_module(name, module):
            attention_names.append(name)
    return attention_names


def count_attention_modules(model: nn.Module) -> int:
    """
    Count the number of attention modules in the model.
    
    Args:
        model: The model to inspect
        
    Returns:
        Number of attention modules
    """
    return len(get_attention_module_names(model))