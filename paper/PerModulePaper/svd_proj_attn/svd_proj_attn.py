r"""
SVD Projection Attention Alias.
Maps to `framework.attentions.svd_projection_attention.SVDProjectionAttention`.
"""

from framework.attentions.svd_projection_attention import (
    SVDProjectionAttention,
    SVDProjectionLinear,
    inject_svd_proj_attn,
)

__all__ = ["SVDProjectionAttention", "SVDProjectionLinear", "inject_svd_proj_attn"]
