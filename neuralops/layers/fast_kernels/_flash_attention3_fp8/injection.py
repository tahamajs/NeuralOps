#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Injection factory for FlashAttention3FP8."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

def inject_flash3_fp8(
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
            res = inject_flash3_fp8(
                layer=lyr,
                model_config=cfg,
                comp_config=(comp_config if "comp_config" in locals() or "comp_config" in kwargs else None),
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target

    r"""Safe injection factory for FlashAttention3FP8."""
    src_attn = old_attn
    if src_attn is None:
        if hasattr(layer, "attention"):
            src_attn = layer.attention
        elif hasattr(layer, "attn"):
            src_attn = layer.attn
        elif hasattr(layer, "self_attn"):
            src_attn = layer.self_attn
        elif hasattr(layer, "self"):
            src_attn = layer.self

    try:
        from neuralops.layers.fast_kernels.flash_attention3_fp8 import FlashAttention3FP8
        new_attn = FlashAttention3FP8(
            config=model_config,
            comp_config=comp_config,
            layer_id=layer_idx,
            layer_idx=layer_idx,
            old_attn=src_attn,
            **kwargs,
        )
        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                layer.attention.self = new_attn
                new_attn.is_nested_encoder = True
            else:
                layer.attention = new_attn
                new_attn.is_nested_encoder = False
        elif hasattr(layer, "self") and hasattr(layer, "output"):
            layer.self = new_attn
            new_attn.is_nested_encoder = True
        elif hasattr(layer, "attn"):
            layer.attn = new_attn
            new_attn.is_nested_encoder = False
        elif hasattr(layer, "self_attn"):
            layer.self_attn = new_attn
            new_attn.is_nested_encoder = False
        return new_attn
    except Exception as e:
        logger.error(f"FlashAttention3FP8 injection failed: {e}")
        return None

