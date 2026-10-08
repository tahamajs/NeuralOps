r"""
Model Attention Replacement and Dynamic Graph Surgery Utilities.

Public API entry points:
  - replace_attention_in_model
  - build_attention
  - inject_replace
  - ReplaceAttention
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from neuralops.core._attention_extractors import (
    _replace_bert_attention,
    _replace_distilbert_attention,
    _replace_gpt_attention,
)
from neuralops.core._attention_helpers import (
    _attach_new_attention,
    _build_attention_from_registry,
    _cfg_get,
    _get_attn_submodule,
    _get_valid_mask,
    _replace_layers,
)
from neuralops.core._attention_wrapper import Replace, ReplaceAttention
from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.common import FactorizedLinear
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.svd_utils import randomized_svd

logger = logging.getLogger(__name__)

# Backward-compatibility aliases
_FactorizedLinear = FactorizedLinear
_randomized_svd = randomized_svd


def build_attention(
    attention_name: str, hidden_size: int, num_heads: int, **kwargs: Any
) -> nn.Module:
    """Public alias for instantiating attention from registry."""
    return _build_attention_from_registry(attention_name, hidden_size, num_heads, **kwargs)


def replace_attention_in_model(
    model: nn.Module,
    attention_name: str,
    attention_params: dict[str, Any] | None = None,
) -> nn.Module:
    """
    Replace all self-attention layers in a transformer model with target attention variant.
    """
    if attention_params is None:
        attention_params = {}

    model_type = getattr(getattr(model, "config", None), "model_type", None)

    if model_type in ("t5", "mt5", "bart", "mbart", "pegasus", "marian"):
        raise NotImplementedError(
            f"replace_attention_in_model does not support encoder-decoder "
            f"models (model_type={model_type}). Cross-attention is not implemented."
        )

    inject_fn = ATTENTION_REGISTRY.get(attention_name)
    if inject_fn is None:
        raise ValueError(f"Attention '{attention_name}' not registered in ATTENTION_REGISTRY.")
    if not callable(inject_fn):
        raise ValueError(f"Attention '{attention_name}' is not callable.")

    cfg = getattr(model, "config", None)

    if model_type in ("bert", "roberta", "xlm-roberta", "electra", "deberta"):
        encoder = getattr(
            getattr(model, model_type.replace("-", "_"), model), "encoder", None
        ) or getattr(getattr(model, "bert", None), "encoder", None)
        if encoder is None:
            raise AttributeError("Could not locate encoder in model architecture.")
        _replace_layers(encoder.layer, "attention", inject_fn, cfg, attention_params, model_type)

    elif model_type == "distilbert":
        distilbert = getattr(model, "distilbert", model)
        layers = distilbert.transformer.layer
        _replace_layers(layers, "attention", inject_fn, cfg, attention_params, model_type)

    elif model_type == "gpt2":
        transformer = getattr(model, "transformer", None)
        if transformer is None:
            raise AttributeError("Could not locate transformer in GPT-2 model.")
        _replace_layers(transformer.h, "attn", inject_fn, cfg, attention_params, model_type)

    elif model_type in (
        "qwen2",
        "qwen3",
        "qwen2_5",
        "llama",
        "mistral",
        "gemma",
        "phi",
        "opt",
        "bloom",
        "falcon",
        "olmo",
        "mpt",
        "stablelm",
    ):
        layers = None
        for path in ("model.layers", "transformer.h", "layers", "decoder.layers"):
            obj = model
            ok = True
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    ok = False
                    break
            if ok:
                layers = obj
                break
        if layers is None:
            raise AttributeError(f"Could not locate decoder layers for model_type={model_type}.")
        _replace_layers(layers, "self_attn", inject_fn, cfg, attention_params, model_type)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model


@ATTENTION_REGISTRY.register("replace_inject")
@ATTENTION_REGISTRY.register("replace_attention_inject")
@ATTENTION_REGISTRY.register("inject_replace")
@ATTENTION_REGISTRY.register("inject_replace_attention")
@ATTENTION_REGISTRY.register("replace")
@ATTENTION_REGISTRY.register("replace_attention")
def inject_replace(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
) -> nn.Module | None:
    """Safe injection of ReplaceAttention with recursion guard."""
    if _depth > 8:
        logger.warning("inject_replace: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # Whole-model traversal
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
    elif (
        hasattr(target, "bert")
        and hasattr(target.bert, "encoder")
        and hasattr(target.bert.encoder, "layer")
    ):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_replace(
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

    # Single-layer injection
    try:
        src = old_attn
        if src is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    if isinstance(cand, nn.Module) and cand is not target:
                        src = getattr(cand, "self", cand)
                        break
        if src is None:
            src = target

        if isinstance(src, ReplaceAttention) or "ReplaceAttention" in src.__class__.__name__:
            return None

        attn_params = {}
        if comp_config is not None:
            for key in ("rank", "dropout"):
                val = getattr(comp_config, key, None)
                if val is not None:
                    attn_params[key] = val

        new_attn = ReplaceAttention(
            config=cfg or getattr(src, "config", None),
            old_attn=src if src is not target else None,
            **attn_params,
            **kwargs,
        )

        _attach_new_attention(target, "attention", new_attn, model_type="")

        return new_attn
    except Exception as e:
        logger.exception("inject_replace failed at layer %d: %s", layer_idx, e)
        return None


inject_replace_attention = inject_replace
inject_replace_attention_v2 = inject_replace

__all__ = [
    "replace_attention_in_model",
    "build_attention",
    "ReplaceAttention",
    "Replace",
    "inject_replace",
    "inject_replace_attention",
    "_replace_bert_attention",
    "_replace_gpt_attention",
    "_replace_distilbert_attention",
    "_get_valid_mask",
    "_cfg_get",
    "_FactorizedLinear",
    "_randomized_svd",
]
