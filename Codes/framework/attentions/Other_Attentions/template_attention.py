"""
Template for creating new attention mechanisms.
Copy and rename this file, then implement your attention class.
Register the injection function with @ATTENTION_REGISTRY.register("your_name").
"""
import torch.nn as nn
from .core.base import BaseAttention
from .core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("your_custom_attention")
def inject_your_attention(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    """Replace the layer's attention with YourCustomAttention."""
    # new_attn = YourCustomAttention(model_config, comp_config, ...)
    # copy weights
    # layer.attention = new_attn
    # return new_attn
    raise NotImplementedError("Implement your injection function here.")
