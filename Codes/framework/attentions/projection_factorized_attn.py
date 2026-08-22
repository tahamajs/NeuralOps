r"""
Projection Factorized Attention Alias.
Maps to `framework.attentions.svd_linear_attention.SVDLinearAttention`.
"""

from framework.attentions.svd_linear_attention import (
    SVDLinear,
    SVDLinearAttention,
    inject_svd_linear,
)

ProjectionFactorizedAttention = SVDLinearAttention
inject_projection_factorized_attn = inject_svd_linear

__all__ = ["ProjectionFactorizedAttention", "SVDLinearAttention", "inject_projection_factorized_attn", "inject_svd_linear"]
