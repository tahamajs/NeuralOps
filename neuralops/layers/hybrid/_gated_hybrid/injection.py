#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Injection factory for GatedHybridKernelAttention."""

from __future__ import annotations

import logging
from typing import Any

import torch.nn as nn

from neuralops.layers.hybrid._gated_hybrid.config import GatedHybridKernelConfig

logger = logging.getLogger(__name__)

def inject_gated_hybrid_kernel(
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
        logger.warning("inject_gated_hybrid_kernel: recursion depth exceeded.")
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
        n_inj = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_gated_hybrid_kernel(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                _depth=_depth + 1,
                **kwargs,
            )
            if res is not None:
                n_inj += 1
        return target if n_inj > 0 else None

    layer = target
    try:

        def _already_injected(m):
            if m is None:
                return False
            if isinstance(m, GatedHybridKernelAttention):
                return True
            if "GatedHybridKernel" in m.__class__.__name__:
                return True
            inner = getattr(m, "inner", None)
            return isinstance(inner, GatedHybridKernelAttention)

        if _already_injected(old_attn):
            return None

        src = old_attn
        if src is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(layer, attr):
                    cand = getattr(layer, attr)
                    if isinstance(cand, nn.Module) and cand is not layer:
                        src = getattr(cand, "self", cand)
                        break
        if src is None:
            return None
        if _already_injected(src):
            return None

        if model_config is None:
            model_config = getattr(layer, "config", None) or getattr(src, "config", None)

        h_size = (
            getattr(
                model_config,
                "hidden_size",
                getattr(model_config, "d_model", getattr(model_config, "n_embd", getattr(model_config, "dim", 768))),
            )
            if model_config is not None
            else 768
        )
        n_heads = (
            getattr(
                model_config,
                "num_attention_heads",
                getattr(
                    model_config, "num_heads", getattr(model_config, "n_heads", getattr(model_config, "n_head", 12))
                ),
            )
            if model_config is not None
            else 12
        )
        h_dim = (
            getattr(model_config, "head_dim", getattr(model_config, "d_kv", h_size // max(1, n_heads)))
            if model_config is not None
            else h_size // max(1, n_heads)
        )
        n_kv_heads = (
            getattr(model_config, "num_key_value_heads", getattr(model_config, "num_kv_heads", n_heads))
            if model_config is not None
            else n_heads
        )

        from neuralops.layers.hybrid.gated_hybrid_kernel_attention import GatedHybridKernelAttention
        is_dec = GatedHybridKernelAttention._detect_decoder(model_config, None, src)

        cfg_obj = GatedHybridKernelConfig(
            hidden_size=h_size,
            num_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            head_dim=h_dim,
            window_size=getattr(comp_config, "window_size", 128) if comp_config else 128,
            svd_rank=getattr(comp_config, "svd_rank", 384) if comp_config else 384,
            low_rank_update=getattr(comp_config, "low_rank_update", 16) if comp_config else 16,
            kernel_feature_dim=getattr(comp_config, "kernel_feature_dim", 128) if comp_config else 128,
            dropout=getattr(comp_config, "dropout", 0.1) if comp_config else 0.1,
            fusion_gate_bias=getattr(comp_config, "fusion_gate_bias", 2.5) if comp_config else 2.5,
            chunk_size=getattr(comp_config, "chunk_size", 128) if comp_config else 128,
            is_decoder=is_dec,
        )

        new_attn = GatedHybridKernelAttention(
            config=cfg_obj,
            layer_idx=layer_idx,
        )
        if src is not None:
            new_attn.copy_weights(src)
            p = next(src.parameters(), None)
            if p is not None:
                new_attn.to(device=p.device, dtype=p.dtype)

        spliced = False
        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                layer.attention.self = new_attn
                new_attn.is_nested_encoder = True
            else:
                layer.attention = new_attn
            spliced = True
        elif hasattr(layer, "self"):
            layer.self = new_attn
            new_attn.is_nested_encoder = True
            spliced = True
        elif hasattr(layer, "attn"):
            layer.attn = new_attn
            spliced = True
        elif hasattr(layer, "self_attn"):
            layer.self_attn = new_attn
            spliced = True

        if not spliced:
            logger.error(
                "inject_gated_hybrid_kernel: no writable attention attribute on %s",
                type(layer).__name__,
            )
            return None

        return new_attn
    except Exception as e:
        logger.exception("inject_gated_hybrid_kernel failed at layer %d: %s", layer_idx, e)
        return None

