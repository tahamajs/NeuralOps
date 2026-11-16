# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Model surgical injection routines for ORION Attention."""

from __future__ import annotations

import logging
from typing import Any

import torch.nn as nn

logger = logging.getLogger(__name__)

def inject_orion(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
) -> ORIONAttention | nn.Module | None:
    from .orion import ORIONAttention
    """Safe injection hook for ORIONAttention across diverse transformer architectures."""
    if _depth > 8:
        logger.warning("inject_orion: max recursion depth exceeded, aborting.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_orion(
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

    try:
        src_attn = old_attn
        if src_attn is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    if cand is None:
                        continue
                    if hasattr(cand, "self"):
                        src_attn = cand.self
                    else:
                        src_attn = cand
                    break

        if src_attn is None and target is not None and hasattr(target, "forward"):
            src_attn = target

        if src_attn is None:
            return None

        if isinstance(src_attn, ORIONAttention):
            return None
        if "ORION" in src_attn.__class__.__name__:
            return None
        for sub_attr in ("inner", "_wrapped_attention"):
            sub = getattr(src_attn, sub_attr, None)
            if isinstance(sub, ORIONAttention):
                return None

        c_name = src_attn.__class__.__name__
        if "DeltaNet" in c_name or "LinearAttention" in c_name:
            return None

        if cfg is None and hasattr(src_attn, "config"):
            cfg = src_attn.config

        hidden_size = (
            kwargs.get("hidden_size")
            or getattr(src_attn, "hidden_size", None)
            or getattr(src_attn, "embed_dim", None)
            or (
                src_attn.q_proj.in_features
                if hasattr(src_attn, "q_proj") and hasattr(src_attn.q_proj, "in_features")
                else None
            )
            or (
                src_attn.query.in_features
                if hasattr(src_attn, "query") and hasattr(src_attn.query, "in_features")
                else None
            )
            or (
                src_attn.q_lin.in_features
                if hasattr(src_attn, "q_lin") and hasattr(src_attn.q_lin, "in_features")
                else None
            )
            or (target.in_features if hasattr(target, "in_features") else None)
            or getattr(cfg, "hidden_size", None)
            or getattr(cfg, "d_model", 768)
        )
        num_heads = (
            kwargs.get("num_heads")
            or kwargs.get("num_attention_heads")
            or getattr(src_attn, "num_heads", None)
            or getattr(src_attn, "num_attention_heads", None)
            or getattr(cfg, "num_attention_heads", None)
            or getattr(cfg, "num_heads", 12)
        )

        new_attn = ORIONAttention(
            config=cfg,
            comp_config=comp_config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_attention_heads=num_heads,
            source_attn=src_attn,
            **kwargs,
        )

        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                target.attention.self = new_attn
                new_attn.is_nested_encoder = True
            else:
                target.attention = new_attn
            return new_attn
        for attr in ("self_attn", "attn", "self"):
            if hasattr(target, attr):
                setattr(target, attr, new_attn)
                return new_attn
        target.attention = new_attn
        return new_attn
    except Exception as e:
        logger.exception("inject_orion failed: %s", e)
        return None


inject_orion_attention = inject_orion

__all__ = [
    "inject_orion",
    "inject_orion_attention",
]
