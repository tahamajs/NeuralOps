r"""
SVD Linear Decay Gated Attention Alias.
Maps to `framework.attentions.svd_linear_attention.SVDLinearAttention`.
"""

from framework.attentions.svd_linear_attention import (
    SVDLinear,
    SVDLinearAttention,
)

DecayGatedLinearAttention = SVDLinearAttention

__all__ = ["DecayGatedLinearAttention", "SVDLinearAttention", "SVDLinear"]
