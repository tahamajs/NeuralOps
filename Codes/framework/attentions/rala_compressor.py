r"""
Rank-Augmented Linear Attention Compressor (RALA Compressor) – Corrected.
Maps to and aliases framework.attentions.rala_attention.RALALinearAttention.

Bug-fix map:
  RALAC-FIX-01  Added missing `import torch` (was used in ModuleList check,
                causing NameError when target.h was a ModuleList).
  RALAC-FIX-02  Removed dead `'comp_config' in locals()` check (always True
                since comp_config is a function parameter).
  RALAC-FIX-03  Recursion guard (`_depth`).
  RALAC-FIX-04  Clean separation of whole-model vs single-layer dispatch.
"""

from typing import Any, Optional

import torch                              # RALAC-FIX-01
import torch.nn as nn

from framework.core.registry import ATTENTION_REGISTRY
from framework.attentions.rala_attention import (
    RALALinearAttention,
    inject_rala,
    inject_rala_attention,
    inject_rala_model,
)


@ATTENTION_REGISTRY.register("rala_compressor")
class RALACompressorAttention(RALALinearAttention):
    r"""Rank-Augmented Linear Attention Compressor."""
    is_custom_attention: bool = True
    pass


def inject_rala_compressor(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    _depth: int = 0,                     # RALAC-FIX-03
    **kwargs: Any,
):
    if _depth > 8:
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer", None))
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
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (torch.nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_rala_compressor(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                _depth=_depth + 1,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    # ---- Single-layer dispatch ----
    if model is not None and target is model:
        return inject_rala_model(
            model, model_config=cfg, comp_config=comp_config, **kwargs,
        )

    return inject_rala(
        layer=target,
        model_config=cfg,
        comp_config=comp_config,
        layer_idx=layer_idx,
        total_layers=total_layers,
        old_attn=old_attn,
        **kwargs,
    )


inject_rala_compressor_attention = inject_rala_compressor

__all__ = [
    "RALALinearAttention",
    "RALACompressorAttention",
    "inject_rala_compressor",
    "inject_rala_compressor_attention",
]