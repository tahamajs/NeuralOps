"""Utility for assigning ranks based on layer patterns."""

import torch
import torch.nn as nn

def get_adaptive_ranks(patterns: list) -> dict:
    """Return a dictionary mapping pattern → rank."""
    ranks = {}
    for pattern in patterns:
        if any(k in pattern.lower() for k in ['out', 'output', 'classifier']):
            ranks[pattern] = 64
        elif any(k in pattern.lower() for k in ['q_lin', 'k_lin', 'v_lin']):
            ranks[pattern] = 32
        elif any(k in pattern.lower() for k in ['query', 'key', 'value']):
            ranks[pattern] = 32
        else:
            ranks[pattern] = 48
    return ranks

def apply_adaptive_svd(model, rank_map):
    """
    Apply SVD compression to linear layers according to pattern-rank mapping.
    Uses SVDLinearLayer from svd_linear.
    """
    from .svd_linear import SVDLinearLayer
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for pattern, rank in rank_map.items():
                if pattern.lower() in name.lower():
                    try:
                        new_layer = SVDLinearLayer(module, rank=rank)
                        parent_name = '.'.join(name.split('.')[:-1])
                        child_name = name.split('.')[-1]
                        parent = model.get_submodule(parent_name) if parent_name else model
                        setattr(parent, child_name, new_layer)
                        count += 1
                    except Exception:
                        pass
                    break   # first match wins
    return count