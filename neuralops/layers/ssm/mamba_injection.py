# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Model surgical injection routines for MambaBlock."""

from __future__ import annotations

import logging
from typing import Any

import torch.nn as nn

from neuralops.core.common import _detect_return_format, _infer_d_model

logger = logging.getLogger(__name__)

def inject_mamba(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
) -> nn.Module | Any | None:
    from .mamba_block import MambaBlock
    if _depth > 8:
        logger.warning("inject_mamba: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

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
            res = inject_mamba(
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

    src_attn = old_attn
    if src_attn is None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    src_attn = getattr(cand, "self", cand)
                    break

    if isinstance(src_attn, MambaBlock):
        return None
    for attr in ("attention", "self_attn", "attn", "self"):
        child = getattr(target, attr, None)
        if isinstance(child, MambaBlock):
            return None

    try:
        d_model = _infer_d_model(cfg, src_attn, None, **kwargs)
    except ValueError as e:
        logger.error("inject_mamba: %s", e)
        return None

    try:
        kwargs.pop("d_model", None)
        kwargs.pop("hidden_size", None)
        kwargs.pop("source_attn", None)
        kwargs.pop("old_attn", None)
        new_module = MambaBlock(
            config=cfg,
            d_model=d_model,
            source_attn=src_attn,
            **kwargs,
        )
    except Exception as e:
        logger.error("MambaBlock instantiation failed: %s", e)
        return None

    if src_attn is not None:
        try:
            new_module.copy_weights(src_attn)
        except Exception as e:
            logger.error("MambaBlock.copy_weights failed: %s", e)
            return None

        p = next(src_attn.parameters(), None)
        if p is not None:
            new_module.to(device=p.device, dtype=p.dtype)

    if target is not None:
        spliced = False
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                new_module.is_nested_encoder = True
                target.attention.self = new_module
            else:
                new_module.is_nested_encoder = False
                target.attention = new_module
            spliced = True
        elif hasattr(target, "self_attn"):
            new_module.is_nested_encoder = False
            target.self_attn = new_module
            spliced = True
        elif hasattr(target, "self"):
            new_module.is_nested_encoder = True
            target.self = new_module
            spliced = True
        elif hasattr(target, "attn"):
            new_module.is_nested_encoder = False
            target.attn = new_module
            spliced = True

        if not spliced:
            logger.error(
                "inject_mamba: no writable attention attribute on %s",
                type(target).__name__,
            )
            return None

        new_module._return_format = _detect_return_format(src_attn, new_module.is_nested_encoder)

    return new_module


__all__ = ["inject_mamba"]
