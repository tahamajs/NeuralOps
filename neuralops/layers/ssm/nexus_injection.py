# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Model surgical injection routines for NEXUS Attention."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from .nexus_components import NEXUSConfig

logger = logging.getLogger(__name__)

def inject_nexus(
    model: nn.Module | None = None,
    layer: nn.Module | None = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
) -> NEXUSAttention | nn.Module | None:
    from .nexus import NEXUSAttention
    if _depth > 8:
        logger.warning("inject_nexus: max recursion depth exceeded.")
        return None

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
            res = inject_nexus(
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
        if src_attn is None and target is not None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    src_attn = getattr(cand, "self", cand)
                    break
        if src_attn is None and target is not None and hasattr(target, "forward"):
            src_attn = target
        if src_attn is None:
            return None

        if isinstance(src_attn, NEXUSAttention) or "NEXUS" in src_attn.__class__.__name__:
            return None

        if cfg is None and hasattr(src_attn, "config"):
            cfg = src_attn.config

        nexus_cfg = kwargs.get("nexus_cfg")
        if nexus_cfg is None:
            nexus_cfg = NEXUSConfig(
                block_size=(getattr(comp_config, "block_size", 16) if comp_config else 16),
                top_k_blocks=(getattr(comp_config, "top_k_inter", 4) if comp_config else 4),
                xattn_enabled=True,
                seer_enabled=True,
                dual_stage_enabled=True,
                stripe_enabled=True,
                rectify_interval=(getattr(comp_config, "rectify_interval", 4) if comp_config else 4),
                rff_features=(getattr(comp_config, "rff_features", 128) if comp_config else 128),
                softpick_enabled=True,
                sparse_min_seq_len=8,
                enable_spectraformer=(getattr(comp_config, "enable_spectraformer", False) if comp_config else False),
                dropout=getattr(comp_config, "dropout", 0.1) if comp_config else 0.1,
                causal_chunk_size=(getattr(comp_config, "causal_chunk_size", 256) if comp_config else 256),
                causal_landmarks=(getattr(comp_config, "causal_landmarks", 32) if comp_config else 32),
            )

        new_attn = NEXUSAttention(
            model_config=cfg,
            nexus_cfg=nexus_cfg,
            layer_idx=layer_idx,
            total_layers=total_layers,
            pretrained_attn=src_attn,
            **kwargs,
        )

        p = next(src_attn.parameters(), None)
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)

        if target is not None:
            if hasattr(target, "attention") and hasattr(target.attention, "self"):
                new_attn.is_nested_encoder = True
                if hasattr(new_attn, "dss_attn"):
                    new_attn.dss_attn.is_nested_encoder = True
                target.attention.self = new_attn
                return new_attn
            if hasattr(target, "attention"):
                new_attn.is_nested_encoder = False
                target.attention = new_attn
                return new_attn
            if hasattr(target, "self_attn"):
                new_attn.is_nested_encoder = False
                target.self_attn = new_attn
                return new_attn
            if hasattr(target, "attn"):
                new_attn.is_nested_encoder = False
                target.attn = new_attn
                return new_attn
            if hasattr(target, "self"):
                new_attn.is_nested_encoder = True
                target.self = new_attn
                return new_attn
            target.attention = new_attn
            return new_attn
        return new_attn
    except Exception as e:
        logger.exception("inject_nexus failed: %s", e)
        return None


inject_nexus_attention = inject_nexus
inject_nexus_into_distilbert = inject_nexus



inject_nexus_attention = inject_nexus
inject_nexus_into_distilbert = inject_nexus

__all__ = [
    "inject_nexus",
    "inject_nexus_attention",
    "inject_nexus_into_distilbert",
]
