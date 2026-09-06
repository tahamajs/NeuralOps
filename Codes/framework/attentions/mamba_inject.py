"""Mamba Injection Module.

Provides injection hooks to wire Bidirectional Mamba into transformer layers
with automated weight copying from pretrained self-attention projections.
"""

from typing import Any, Optional

import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("mamba_module_inject")
def inject_mamba(
    layer: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 6,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Any:
    r"""Injection hook that instantiates Mamba and wires it into a transformer layer."""
    from framework.attentions.mamba import Mamba

    rank = kwargs.pop("rank", 32)
    d_model = 128
    if model_config is not None:
        d_model = getattr(model_config, "hidden_size", getattr(model_config, "dim", getattr(model_config, "d_model", 128)))
    elif layer is not None and hasattr(layer, "config"):
        d_model = getattr(layer.config, "hidden_size", getattr(layer.config, "dim", 128))
    elif layer is not None and hasattr(layer, "hidden_size"):
        d_model = getattr(layer, "hidden_size", 128)

    new_module = Mamba(d_model=d_model, config=model_config, rank=rank, **kwargs)

    src_attn = old_attn if old_attn is not None else layer
    if src_attn is not None and hasattr(new_module, "copy_weights"):
        try:
            new_module.copy_weights(src_attn)
        except Exception:
            pass

    if layer is not None:
        if hasattr(layer, "attention"):
            layer.attention = new_module
        elif hasattr(layer, "self_attn"):
            layer.self_attn = new_module
        elif hasattr(layer, "self"):
            layer.self = new_module
        else:
            layer.attention = new_module

    return new_module
