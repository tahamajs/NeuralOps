r"""
Rank-Augmented Linear Attention Compressor (RALA Compressor) – Production-Hardened.
Maps to and aliases framework.attentions.rala_attention.RALALinearAttention.
"""

from typing import Any, Optional
import torch.nn as nn

from framework.core.registry import ATTENTION_REGISTRY
from framework.attentions.rala_attention import (
    RALALinearAttention,
    inject_rala,
    inject_rala_attention,
    inject_rala_model,
)

@ATTENTION_REGISTRY.register("rala_compressor")
class RALACompressorAttention(RALALinearAttention):
    r"""Rank-Augmented Linear Attention Compressor."""
    pass


def inject_rala_compressor(
    layer: Optional[nn.Module] = None,
    model: Optional[nn.Module] = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[nn.Module]:
    if model is not None:
        return inject_rala_model(model, **kwargs)
    return inject_rala(
        layer=layer,
        model_config=model_config,
        comp_config=comp_config,
        layer_idx=layer_idx,
        total_layers=total_layers,
        old_attn=old_attn,
        **kwargs,
    )


inject_rala_compressor_attention = inject_rala_compressor

__all__ = [
    "RALALinearAttention",
    "RALACompressorAttention",
    "inject_rala_compressor",
    "inject_rala_compressor_attention",
]
