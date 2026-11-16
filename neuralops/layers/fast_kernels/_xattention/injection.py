#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Injection factory for xAttention."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from neuralops.core.common import _cfg_get

logger = logging.getLogger(__name__)

def inject_xattention(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
):
    if _depth > 8:
        logger.warning("inject_xattention: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    layer_list = None
    for path in (
        ("distilbert", "transformer", "layer"),
        ("roberta", "encoder", "layer"),
        ("bert", "encoder", "layer"),
        ("transformer", "h"),
        ("model", "layers"),
        ("encoder", "layer"),
        ("layers",),
        ("h",),
    ):
        obj = target
        ok = True
        for p in path:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False
                break
        if ok and isinstance(obj, (nn.ModuleList, list)):
            layer_list = obj
            break

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_xattention(
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

    from neuralops.layers.fast_kernels.xattention import XAttention

    src_attn = old_attn
    if src_attn is None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    src_attn = getattr(cand, "self", cand)
                    break

    if src_attn is None:
        logger.warning(
            "inject_xattention: no source attention found on %s",
            type(target).__name__,
        )
        return None

    if isinstance(src_attn, XAttention) or "XAttention" in src_attn.__class__.__name__:
        return None

    if cfg is None:
        cfg = getattr(src_attn, "config", None) or getattr(target, "config", None)

    hidden_size = (
        _cfg_get(cfg, "hidden_size")
        or _cfg_get(cfg, "d_model")
        or _cfg_get(cfg, "dim")
        or getattr(src_attn, "hidden_size", None)
        or getattr(src_attn, "embed_dim", None)
        or (
            src_attn.q_proj.in_features
            if hasattr(src_attn, "q_proj") and hasattr(src_attn.q_proj, "in_features")
            else None
        )
        or (
            src_attn.q_lin.in_features
            if hasattr(src_attn, "q_lin") and hasattr(src_attn.q_lin, "in_features")
            else None
        )
        or (
            src_attn.query.in_features
            if hasattr(src_attn, "query") and hasattr(src_attn.query, "in_features")
            else None
        )
        or 768
    )
    num_heads = (
        _cfg_get(cfg, "num_attention_heads")
        or _cfg_get(cfg, "num_heads")
        or _cfg_get(cfg, "n_heads")
        or getattr(src_attn, "num_heads", None)
        or getattr(src_attn, "num_attention_heads", None)
        or 12
    )

    if cfg is None:
        import types

        cfg = types.SimpleNamespace(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=getattr(src_attn, "num_key_value_heads", num_heads),
            is_decoder=getattr(src_attn, "is_decoder", False),
        )

    lora_rank = kwargs.pop("lora_rank", _cfg_get(comp_config, "lora_rank", 16))
    rank = kwargs.pop("rank", _cfg_get(comp_config, "rank", 32))
    block_size = kwargs.pop("block_size", _cfg_get(comp_config, "block_size", 32))
    sparsity = kwargs.pop("sparsity", _cfg_get(comp_config, "sparsity", 0.5))

    try:
        new_attn = XAttention(
            config=cfg,
            rank=rank,
            block_size=block_size,
            sparsity=sparsity,
            lora_rank=lora_rank,
            pretrained_attn=src_attn,
            layer_idx=layer_idx,
            **kwargs,
        )

        p = next(src_attn.parameters(), None)
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)

        spliced = False
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                target.attention.self = new_attn
                new_attn.is_nested_encoder = True
            else:
                target.attention = new_attn
                new_attn.is_nested_encoder = False
            spliced = True
        elif hasattr(target, "self_attn"):
            target.self_attn = new_attn
            spliced = True
        elif hasattr(target, "attn"):
            target.attn = new_attn
            spliced = True
        elif hasattr(target, "self"):
            target.self = new_attn
            new_attn.is_nested_encoder = True
            spliced = True

        if not spliced:
            logger.error(
                "inject_xattention: no writable attention attribute on %s",
                type(target).__name__,
            )
            return None

        return new_attn

    except Exception as e:
        logger.exception("inject_xattention failed at layer %d: %s", layer_idx, e)
        return None

