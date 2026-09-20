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
    is_custom_attention: bool = True
    pass


def inject_projection_factorized_attn(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
):
    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer", None))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # Whole-model traversal
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "h") and isinstance(target.h, (torch.nn.ModuleList, list)):
        layer_list = target.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (torch.nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_projection_factorized_attn(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config if 'comp_config' in locals() or 'comp_config' in kwargs else None,
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target

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
