r"""
SVD Linear Decay Gated Attention Alias.
Maps to `framework.attentions.svd_linear_attention.SVDLinearAttention`.
"""

from framework.core.registry import ATTENTION_REGISTRY
from framework.attentions.svd_linear_attention import (
    SVDLinear,
    SVDLinearAttention,
)

@ATTENTION_REGISTRY.register("svd_linear_decay")
@ATTENTION_REGISTRY.register("decay_gated_linear_attention")
class DecayGatedLinearAttention(SVDLinearAttention):
    r"""Decay Gated SVD Linear Attention."""
    pass

__all__ = ["DecayGatedLinearAttention", "SVDLinearAttention", "SVDLinear"]
