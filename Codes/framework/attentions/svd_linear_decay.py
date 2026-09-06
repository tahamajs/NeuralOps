r"""
SVD Linear Decay Gated Attention Alias.
Maps to `framework.attentions.svd_linear_attention.SVDLinearAttention`.
"""

from typing import Any, Optional
import torch.nn as nn

from framework.core.registry import ATTENTION_REGISTRY
from framework.attentions.svd_linear_attention import (
    SVDLinear,
    SVDLinearAttention,
    inject_svd_linear_attention,
)

@ATTENTION_REGISTRY.register("svd_linear_decay")
@ATTENTION_REGISTRY.register("decay_gated_linear_attention")
class DecayGatedLinearAttention(SVDLinearAttention):
    r"""Decay Gated SVD Linear Attention."""
    def __init__(self, *args, **kwargs):
        kwargs["use_decay_gating"] = True
        super().__init__(*args, **kwargs)


def inject_svd_linear_decay(
    model: Optional[nn.Module] = None,
    layer: Optional[nn.Module] = None,
    model_config: Optional[Any] = None,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[nn.Module]:
    kwargs["use_decay_gating"] = True
    return inject_svd_linear_attention(
        model=model,
        layer=layer,
        model_config=model_config,
        old_attn=old_attn,
        **kwargs,
    )


inject_svd_linear_decay_attention = inject_svd_linear_decay

__all__ = [
    "DecayGatedLinearAttention",
    "SVDLinearAttention",
    "SVDLinear",
    "inject_svd_linear_decay",
    "inject_svd_linear_decay_attention",
]
