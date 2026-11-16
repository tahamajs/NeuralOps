# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Model surgical injection routines for VSpec Attention."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from neuralops.core.common import _cfg_get, _infer_d_model

logger = logging.getLogger(__name__)

def inject_vspec_attn(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    **kwargs: Any,
) -> VSpecAttention | nn.Module | None:
    from .vspec import VSpecAttention
    """Safe injection hook for VSpecAttention across diverse transformer architectures."""
    target = layer if layer is not None else model
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
            res = inject_vspec_attn(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    try:
        src = old_attn
        if src is None and target is not None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    if hasattr(cand, "self"):
                        src = cand.self
                    else:
                        src = cand
                    break

        if src is None and target is not None and hasattr(target, "forward"):
            src = target

        if src is None:
            return None

        if src is not None:
            if isinstance(src, VSpecAttention):
                return None
            if "VSpec" in src.__class__.__name__:
                return None
            if hasattr(src, "inner") and isinstance(src.inner, VSpecAttention):
                return None
            if hasattr(src, "_wrapped_attention") and isinstance(src._wrapped_attention, VSpecAttention):
                return None

        if target is not None:
            for attr in ("attention", "self_attn", "attn", "self"):
                cand = getattr(target, attr, None)
                if cand is None:
                    continue
                if isinstance(cand, VSpecAttention):
                    return None
                if "VSpec" in cand.__class__.__name__:
                    return None
                if hasattr(cand, "inner") and isinstance(cand.inner, VSpecAttention):
                    return None
                if hasattr(cand, "_wrapped_attention") and isinstance(cand._wrapped_attention, VSpecAttention):
                    return None

        if src is not None:
            c_name = src.__class__.__name__
            if "DeltaNet" in c_name or "LinearAttention" in c_name:
                return None

        if cfg is None and hasattr(src, "config"):
            cfg = src.config

        (
            _cfg_get(cfg, "hidden_size")
            or _cfg_get(cfg, "d_model")
            or _cfg_get(cfg, "dim")
            or getattr(src, "hidden_size", None)
            or getattr(src, "embed_dim", None)
            or (src.q_proj.in_features if hasattr(src, "q_proj") and hasattr(src.q_proj, "in_features") else None)
            or (src.q_lin.in_features if hasattr(src, "q_lin") and hasattr(src.q_lin, "in_features") else None)
            or (src.query.in_features if hasattr(src, "query") and hasattr(src.query, "in_features") else None)
            or 768
        )
        (
            _cfg_get(cfg, "num_attention_heads")
            or _cfg_get(cfg, "num_heads")
            or _cfg_get(cfg, "n_heads")
            or getattr(src, "num_heads", None)
            or getattr(src, "num_attention_heads", None)
            or 12
        )

        rank = kwargs.pop(
            "rank",
            _cfg_get(comp_config, "vspec_rank", _cfg_get(comp_config, "rank", 64)),
        )
        lora_rank = kwargs.pop("lora_rank", _cfg_get(comp_config, "lora_rank", 16))

        new_block = VSpecAttention(
            config=cfg,
            rank=rank,
            lora_rank=lora_rank,
            pretrained_attn=src,
            **kwargs,
        )

        if target is not None:
            if hasattr(target, "attention"):
                if hasattr(target.attention, "self"):
                    target.attention.self = new_block
                    new_block.is_nested_encoder = True
                else:
                    target.attention = new_block
                    new_block.is_nested_encoder = False
                return new_block
            for attr in ("self_attn", "attn"):
                if hasattr(target, attr):
                    setattr(target, attr, new_block)
                    new_block.is_nested_encoder = False
                    return new_block
            if hasattr(target, "self"):
                target.self = new_block
                new_block.is_nested_encoder = True
                return new_block
            target.attention = new_block
            new_block.is_nested_encoder = False
            return new_block
        return new_block
    except Exception as e:
        logger.error("inject_vspec_attn failed at layer %d: %s", layer_idx, e)
        return None


inject_vspec = inject_vspec_attn

__all__ = [
    "inject_vspec_attn",
    "inject_vspec",
]
