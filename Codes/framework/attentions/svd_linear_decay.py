r"""
SVD Linear Decay Gated Attention Alias.
Maps to `framework.attentions.svd_linear_attention.SVDLinearAttention`.
"""

from framework.attentions.svd_linear_attention import (
    SVDLinear,
    SVDLinearAttention,
    inject_svd_linear_decay,
)

DecayGatedLinearAttention = SVDLinearAttention

__all__ = ["DecayGatedLinearAttention", "SVDLinearAttention", "inject_svd_linear_decay"]
