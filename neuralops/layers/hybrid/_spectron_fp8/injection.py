#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Injection factory for Aegis-Spectron-FP8 Hybrid Attention."""

import logging
from typing import Any

import torch.nn as nn

from neuralops.core.common import _detect_decoder
from neuralops.layers.hybrid._spectron_fp8.kernels import AegisSpectronFP8Config

logger = logging.getLogger(__name__)

def inject_aegis_spectron_fp8(
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
        logger.warning("inject_aegis_spectron_fp8: max recursion depth.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg_root = model_config if model_config is not None else getattr(target, "config", None)

    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "h") and isinstance(target.h, (nn.ModuleList, list)):
        layer_list = target.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        n_inj = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_aegis_spectron_fp8(
                layer=lyr,
                model_config=cfg_root,
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
        if old_attn is None:
            for attr in ("attention", "attn", "self_attn", "self"):
                if hasattr(layer, attr):
                    cand = getattr(layer, attr)
                    if cand is not None:
                        old_attn = getattr(cand, "self", cand)
                        break

        if model_config is None:
            if any(hasattr(layer, a) for a in ("hidden_size", "d_model", "n_embd", "dim")):
                model_config = layer
            elif old_attn is not None:
                model_config = getattr(old_attn, "config", None)

        h_size = (
            getattr(
                model_config,
                "hidden_size",
                getattr(
                    model_config,
                    "d_model",
                    getattr(model_config, "n_embd", getattr(model_config, "dim", 768)),
                ),
            )
            if model_config is not None
            else 768
        )
        n_heads = (
            getattr(
                model_config,
                "num_attention_heads",
                getattr(
                    model_config,
                    "num_heads",
                    getattr(model_config, "n_heads", getattr(model_config, "n_head", 12)),
                ),
            )
            if model_config is not None
            else 12
        )
        h_dim = (
            getattr(
                model_config,
                "head_dim",
                getattr(model_config, "kv_channels", h_size // max(1, n_heads)),
            )
            if model_config is not None
            else h_size // max(1, n_heads)
        )
        n_kv_heads = (
            getattr(
                model_config,
                "num_key_value_heads",
                getattr(model_config, "num_kv_heads", n_heads),
            )
            if model_config is not None
            else n_heads
        )

        is_dec = _detect_decoder(model_config, kwargs.get("is_decoder"), old_attn)

        cfg = AegisSpectronFP8Config(
            hidden_size=h_size,
            num_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            head_dim=h_dim,
            window_size=(
                getattr(comp_config, "window_size", getattr(comp_config, "local_window", 64)) if comp_config else 64
            ),
            svd_rank=(getattr(comp_config, "svd_rank", getattr(comp_config, "rank", 128)) if comp_config else 128),
            lora_rank=(getattr(comp_config, "lora_rank", getattr(comp_config, "lora_r", 32)) if comp_config else 32),
            dropout=getattr(comp_config, "dropout", 0.1) if comp_config else 0.1,
            is_decoder=is_dec,
            fp8_format=(getattr(comp_config, "fp8_format", "e4m3") if comp_config else "e4m3"),
            chunk_size=getattr(comp_config, "chunk_size", 64) if comp_config else 64,
        )

        try:
            from neuralops.layers.hybrid.spectron_fp8_attention import AegisSpectronFP8Attention

            new_attn = AegisSpectronFP8Attention(config=cfg, pretrained_attn=old_attn)
        except Exception as e:
            logger.error("inject_aegis_spectron_fp8: construction failed: %s", e)
            return None

        if old_attn is not None and hasattr(old_attn, "rotary_emb"):
            new_attn.rotary_emb = old_attn.rotary_emb

        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                new_attn.is_nested_encoder = True
                layer.attention.self = new_attn
            else:
                new_attn.is_nested_encoder = False
                layer.attention = new_attn
        elif hasattr(layer, "self_attn"):
            new_attn.is_nested_encoder = False
            layer.self_attn = new_attn
        elif hasattr(layer, "self"):
            new_attn.is_nested_encoder = False
            layer.self = new_attn
        elif hasattr(layer, "attn"):
            new_attn.is_nested_encoder = False
            layer.attn = new_attn
        return new_attn
    except Exception as e:
        logger.exception("Failed to inject AEGIS-Spectron-FP8: %s", e)
        return None

