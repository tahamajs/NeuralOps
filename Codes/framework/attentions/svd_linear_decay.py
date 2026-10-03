"""
SVD Linear Decay Gated Attention Alias.
Maps to `framework.attentions.svd_linear_attention.SVDLinearAttention`.

"""

import logging
from typing import Any

import torch
import torch.nn as nn
from framework.attentions.svd_linear_attention import (
    SVDLinear,
    SVDLinearAttention,
    inject_svd_linear_attention,
)
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register("svd_linear_decay")
@ATTENTION_REGISTRY.register("decay_gated_linear_attention")
class DecayGatedLinearAttention(SVDLinearAttention):
    r"""Decay Gated SVD Linear Attention."""

    is_custom_attention: bool = True

    def __init__(self, *args, **kwargs):
        kwargs["use_decay_gating"] = True
        super().__init__(*args, **kwargs)


def inject_svd_linear_decay(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    **kwargs: Any,
):
    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # ---- Whole-model traversal ----
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
    elif (
        hasattr(target, "bert")
        and hasattr(target.bert, "encoder")
        and hasattr(target.bert.encoder, "layer")
    ):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(
        target.layers, (torch.nn.ModuleList, list)
    ):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_svd_linear_decay(
                layer=lyr,
                model_config=cfg,
                comp_config=(
                    comp_config
                    if "comp_config" in locals() or "comp_config" in kwargs
                    else None
                ),
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    # ---- Single-layer injection ----
    layer = target

    # ============================================================
    # FIX-1: Discover source attention across all architecture variants
    # ============================================================
    src_attn = old_attn
    src_attr_name: str | None = None
    src_is_nested: bool = False

    if src_attn is None:
        for attr in ("attn", "attention", "self_attn", "self"):
            candidate = getattr(layer, attr, None)
            if candidate is None:
                continue

            # FIX-4: idempotency guard
            if (
                isinstance(candidate, SVDLinearAttention)
                or "SVDLinear" in candidate.__class__.__name__
            ):
                logger.debug(
                    "inject_svd_linear_decay: %s.%s is already SVDLinearAttention, skipping",
                    type(layer).__name__,
                    attr,
                )
                return None

            # RoBERTa/BERT nested: layer.attention.self has .query
            if hasattr(candidate, "self") and hasattr(candidate.self, "query"):
                src_attn = candidate.self
                src_attr_name = attr
                src_is_nested = True
            else:
                src_attn = candidate
                src_attr_name = attr
                src_is_nested = False
            break

    if src_attn is None:
        # FIX-5: explicit error logging
        logger.warning(
            "inject_svd_linear_decay: no source attention found on %s "
            "(checked: attn, attention, self_attn, self)",
            type(layer).__name__,
        )
        return None

    # ============================================================
    # Delegate to underlying injector with explicit old_attn
    # (so it never has to guess the source attribute)
    # ============================================================
    kwargs["use_decay_gating"] = True
    try:
        result = inject_svd_linear_attention(
            model=model,
            layer=layer,
            model_config=model_config,
            old_attn=src_attn,
            **kwargs,
        )
    except Exception as e:
        logger.error("inject_svd_linear_attention delegate failed: %s", e)
        return None

    if result is None:
        return None

    # ============================================================
    # FIX-3: Fallback replacement for GPT-2 and other missed architectures
    # ============================================================
    if src_attr_name is not None and result is not layer:
        if src_is_nested:
            parent = getattr(layer, src_attr_name, None)
            if parent is not None and getattr(parent, "self", None) is src_attn:
                parent.self = result
                logger.debug(
                    "inject_svd_linear_decay: forced replacement of %s.%s.self",
                    type(layer).__name__,
                    src_attr_name,
                )
        else:
            if getattr(layer, src_attr_name, None) is src_attn:
                setattr(layer, src_attr_name, result)
                logger.debug(
                    "inject_svd_linear_decay: forced replacement of %s.%s",
                    type(layer).__name__,
                    src_attr_name,
                )

    return result


inject_svd_linear_decay_attention = inject_svd_linear_decay

__all__ = [
    "DecayGatedLinearAttention",
    "SVDLinearAttention",
    "SVDLinear",
    "inject_svd_linear_decay",
    "inject_svd_linear_decay_attention",
]
