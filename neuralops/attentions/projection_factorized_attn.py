"""Projection Factorized SVD Linear Attention — Production-Hardened (v2).

Delegates to `inject_svd_linear_attention` with a pre-discovered source
attention and robust attribute-level replacement for all supported
architectures (GPT-2, RoBERTa/BERT, DistilBERT, T5, generic).

"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from neuralops.attentions.svd_linear_attention import (
    SVDLinear,
    SVDLinearAttention,
    inject_svd_linear_attention,
)
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


# ===========================================================================
# Registered class
# ===========================================================================
@ATTENTION_REGISTRY.register("projection_factorized_attn")
class ProjectionFactorizedAttention(SVDLinearAttention):
    r"""Projection Factorized SVD Linear Attention."""

    is_custom_attention: bool = True


# ===========================================================================
# Helpers
# ===========================================================================
_CANDIDATE_ATTRS = ("attn", "attention", "self_attn", "self")

# Keys that must NOT be passed through `**kwargs` to the delegate, since
# they are positional/keyword arguments of the delegate signature itself.
_RESERVED_KEYS = frozenset(
    {
        "model",
        "layer",
        "model_config",
        "comp_config",
        "old_attn",
        "layer_idx",
        "total_layers",
    }
)


def _is_already_injected(module: nn.Module) -> bool:
    """True if `module` is an SVDLinearAttention (or subclass)."""
    if module is None:
        return False
    if isinstance(module, SVDLinearAttention):
        return True
    if isinstance(module, ProjectionFactorizedAttention):
        return True
    return "SVDLinearAttention" in module.__class__.__name__


def _find_source_attention(
    layer: nn.Module,
) -> tuple[str, nn.Module, bool] | None:
    """
    Return (attr_name, src_attn, src_is_nested) if a non-injected source
    attention is found; otherwise None.

    Scans `_CANDIDATE_ATTRS` in priority order and skips attributes that
    already contain an injected attention (FIX-10).
    """
    for attr in _CANDIDATE_ATTRS:
        candidate = getattr(layer, attr, None)
        if candidate is None:
            continue

        # FIX-10: skip already-injected attributes, keep scanning others.
        if _is_already_injected(candidate):
            # Nested case: layer.attention.self may be injected.
            if hasattr(candidate, "self") and _is_already_injected(candidate.self):
                continue
            if _is_already_injected(candidate):
                continue

        # RoBERTa/BERT nested: layer.attention.self has .query
        if hasattr(candidate, "self") and hasattr(candidate.self, "query"):
            return attr, candidate.self, True

        return attr, candidate, False

    return None


def _attach_with_fallback(
    layer: nn.Module,
    attr_name: str,
    src_attn: nn.Module,
    src_is_nested: bool,
    result: nn.Module,
) -> bool:
    """
    FIX-3 / FIX-9: Ensure `result` is attached to the correct attribute slot.

    Returns True if a forced replacement was performed, False otherwise.
    """
    if result is None or result is layer or result is src_attn:
        return False
    if not isinstance(result, nn.Module):
        return False

    if src_is_nested:
        parent = getattr(layer, attr_name, None)
        if parent is not None and getattr(parent, "self", None) is src_attn:
            parent.self = result
            logger.debug(
                "inject_projection_factorized_attn: forced replacement of %s.%s.self",
                type(layer).__name__,
                attr_name,
            )
            return True
    else:
        if getattr(layer, attr_name, None) is src_attn:
            setattr(layer, attr_name, result)
            logger.debug(
                "inject_projection_factorized_attn: forced replacement of %s.%s",
                type(layer).__name__,
                attr_name,
            )
            return True

    return False


# ===========================================================================
# Injector
# ===========================================================================
def inject_projection_factorized_attn(
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
    elif hasattr(target, "layers") and isinstance(target.layers, (torch.nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        # FIX-8: strip reserved keys once (they'll be re-supplied explicitly).
        forward_kwargs = {k: v for k, v in kwargs.items() if k not in _RESERVED_KEYS}
        for idx, lyr in enumerate(layer_list):
            res = inject_projection_factorized_attn(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,  # FIX-6
                layer_idx=idx,
                total_layers=tot,  # FIX-11
                **forward_kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    # ---- Single-layer injection ----
    layer = target

    # FIX-1: discover source attention
    src_info = _find_source_attention(layer) if old_attn is None else None
    if old_attn is not None:
        src_attn = old_attn
        src_attr_name = None
        src_is_nested = False
    elif src_info is not None:
        src_attr_name, src_attn, src_is_nested = src_info
    else:
        logger.warning(
            "inject_projection_factorized_attn: no source attention found on %s (checked: %s)",
            type(layer).__name__,
            ", ".join(_CANDIDATE_ATTRS),
        )
        return None

    # FIX-4 / FIX-10: already-injected guard for the specific src_attn
    if _is_already_injected(src_attn):
        logger.debug(
            "inject_projection_factorized_attn: %s is already injected, skipping",
            type(src_attn).__name__,
        )
        return None

    # FIX-8: clean kwargs before delegating.
    delegate_kwargs = {k: v for k, v in kwargs.items() if k not in _RESERVED_KEYS}

    # FIX-2 / FIX-6 / FIX-11: delegate with explicit old_attn + comp_config
    try:
        result = inject_svd_linear_attention(
            model=model,
            layer=layer,
            model_config=model_config if model_config is not None else cfg,
            comp_config=comp_config,  # FIX-6
            old_attn=src_attn,  # FIX-2
            layer_idx=layer_idx,
            total_layers=total_layers,  # FIX-11
            **delegate_kwargs,
        )
    except Exception as e:
        logger.error(
            "inject_svd_linear_attention delegate failed on %s: %s",
            type(layer).__name__,
            e,
        )
        return None

    if result is None:
        return None

    # FIX-3 / FIX-9: fallback attribute replacement
    if src_attr_name is not None:
        _attach_with_fallback(
            layer=layer,
            attr_name=src_attr_name,
            src_attn=src_attn,
            src_is_nested=src_is_nested,
            result=result,
        )

    return result


inject_projection_factorized_attention = inject_projection_factorized_attn

__all__ = [
    "ProjectionFactorizedAttention",
    "SVDLinearAttention",
    "SVDLinear",
    "inject_projection_factorized_attn",
    "inject_projection_factorized_attention",
]
