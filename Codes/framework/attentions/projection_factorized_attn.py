r"""
Projection Factorized Attention Alias.
Maps to `framework.attentions.svd_linear_attention.SVDLinearAttention`.
"""

from framework.attentions.svd_linear_attention import (
    SVDLinear,
    SVDLinearAttention,
)

ProjectionFactorizedAttention = SVDLinearAttention

__all__ = ["ProjectionFactorizedAttention", "SVDLinearAttention", "SVDLinear"]
