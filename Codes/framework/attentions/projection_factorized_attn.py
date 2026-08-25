r"""
Projection Factorized Attention Alias.
Maps to `framework.attentions.svd_linear_attention.SVDLinearAttention`.
"""

from framework.core.registry import ATTENTION_REGISTRY
from framework.attentions.svd_linear_attention import (
    SVDLinear,
    SVDLinearAttention,
)

@ATTENTION_REGISTRY.register("projection_factorized_attn")
class ProjectionFactorizedAttention(SVDLinearAttention):
    r"""Projection Factorized SVD Linear Attention."""
    pass

__all__ = ["ProjectionFactorizedAttention", "SVDLinearAttention", "SVDLinear"]
