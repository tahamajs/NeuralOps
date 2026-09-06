from typing import Any, Optional
import torch.nn as nn

from framework.core.registry import ATTENTION_REGISTRY
from framework.attentions.svd_linear_attention import (
    SVDLinear,
    SVDLinearAttention,
    inject_svd_linear_attention,
)

@ATTENTION_REGISTRY.register("projection_factorized_attn")
class ProjectionFactorizedAttention(SVDLinearAttention):
    r"""Projection Factorized SVD Linear Attention."""
    pass


def inject_projection_factorized_attn(
    model: Optional[nn.Module] = None,
    layer: Optional[nn.Module] = None,
    model_config: Optional[Any] = None,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[nn.Module]:
    return inject_svd_linear_attention(
        model=model,
        layer=layer,
        model_config=model_config,
        old_attn=old_attn,
        **kwargs,
    )


inject_projection_factorized_attention = inject_projection_factorized_attn

__all__ = [
    "ProjectionFactorizedAttention",
    "SVDLinearAttention",
    "SVDLinear",
    "inject_projection_factorized_attn",
    "inject_projection_factorized_attention",
]
