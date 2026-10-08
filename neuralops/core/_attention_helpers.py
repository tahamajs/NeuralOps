"""
Module tree traversal and attention substitution helpers for NeuralOps.
"""

from __future__ import annotations

import inspect
import logging
from argparse import Namespace
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.common import FactorizedLinear, LoRALinear, _cfg_get
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.svd_utils import randomized_svd

logger = logging.getLogger(__name__)

# Backward-compatibility alias
_get_valid_mask = get_token_valid_mask


def _build_attention_from_registry(
    attention_name: str,
    hidden_size: int,
    num_heads: int,
    **kwargs: Any,
) -> nn.Module:
    """Instantiate an attention module from the registry with signature introspection."""
    attn_cls = ATTENTION_REGISTRY.get(attention_name)
    if attn_cls is None:
        raise ValueError(f"Attention '{attention_name}' not registered in ATTENTION_REGISTRY.")
    if not callable(attn_cls):
        raise ValueError(f"Attention '{attention_name}' is not callable.")

    try:
        sig = inspect.signature(attn_cls.__init__)
        params = sig.parameters
    except (ValueError, TypeError):
        params = {}

    kwargs_clean = dict(kwargs)
    cfg_val = kwargs_clean.pop("config", None)
    kwargs_clean.pop("hidden_size", None)
    kwargs_clean.pop("num_heads", None)
    kwargs_clean.pop("num_attention_heads", None)

    attempts = []

    if "config" in params:
        ns = Namespace(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_heads=num_heads,
            head_dim=hidden_size // max(1, num_heads),
            **kwargs_clean,
        )
        if cfg_val is not None:
            attempts.append(lambda: attn_cls(config=cfg_val, **kwargs_clean))
        attempts.append(lambda: attn_cls(config=ns, **kwargs_clean))

    if "num_attention_heads" in params:
        attempts.append(
            lambda: attn_cls(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                **kwargs_clean,
            )
        )

    if "num_heads" in params:
        attempts.append(
            lambda: attn_cls(
                hidden_size=hidden_size,
                num_heads=num_heads,
                **kwargs_clean,
            )
        )

    attempts.append(lambda: attn_cls(hidden_size, num_heads, **kwargs_clean))
    attempts.append(lambda: attn_cls(**kwargs_clean))

    last_exc: Exception | None = None
    for fn in attempts:
        try:
            return fn()
        except Exception as e:
            last_exc = e
            continue

    raise ValueError(
        f"Failed to instantiate '{attention_name}' with hidden_size={hidden_size}, "
        f"num_heads={num_heads}. Last error: {last_exc}"
    )


def _get_attn_submodule(layer: nn.Module, attn_attr: str, model_type: str = "") -> nn.Module | None:
    """Get the innermost attention submodule (unwrap nested BERT/RoBERTa)."""
    container = getattr(layer, attn_attr, None)
    if container is None:
        for fallback in ("attention", "self_attn", "attn", "self"):
            if hasattr(layer, fallback):
                container = getattr(layer, fallback)
                break
    if container is None:
        return None

    if hasattr(container, "self"):
        return container.self
    return container


def _attach_new_attention(
    layer: nn.Module, attn_attr: str, new_attn: nn.Module | None, model_type: str = ""
) -> None:
    """Attach new attention module with proper guard for nested encoders."""
    if new_attn is None:
        logger.error(
            "Cannot attach None attention to layer %s (attr=%s)",
            type(layer).__name__,
            attn_attr,
        )
        return
    container = getattr(layer, attn_attr, None)
    if container is not None and hasattr(container, "self"):
        container.self = new_attn
        return

    if hasattr(layer, "attention"):
        if hasattr(layer.attention, "self"):
            layer.attention.self = new_attn
        else:
            layer.attention = new_attn
        return

    if hasattr(layer, "self_attn"):
        layer.self_attn = new_attn
        return

    if hasattr(layer, "attn"):
        layer.attn = new_attn
        return

    if hasattr(layer, "self"):
        layer.self = new_attn


def _replace_layers(
    layers: list[nn.Module],
    attn_attr: str,
    inject_fn: Any,
    model_config: Any,
    attention_params: dict[str, Any],
    model_type: str,
) -> None:
    """
    Replace attention in every layer of `layers`.

    Tries the registry injector first; falls back to _build_attention_from_registry.
    """
    from neuralops.core._attention_extractors import (
        _infer_hidden_size,
        _infer_num_heads,
        _resolve_class_name,
    )

    total_layers = len(layers)

    try:
        sig = inspect.signature(inject_fn)
        inject_supports_layer_idx = "layer_idx" in sig.parameters
    except (ValueError, TypeError):
        inject_supports_layer_idx = False

    comp_config_obj = SimpleNamespace(**attention_params) if attention_params else None

    SAFE_ATTN_KEYS = {
        "rank",
        "dropout",
        "num_heads",
        "num_attention_heads",
        "hidden_size",
        "num_key_value_heads",
        "is_decoder",
    }
    safe_params = {k: v for k, v in (attention_params or {}).items() if k in SAFE_ATTN_KEYS}

    for idx, layer in enumerate(layers):
        old_attn = _get_attn_submodule(layer, attn_attr, model_type)

        if inject_supports_layer_idx and old_attn is not None:
            try:
                new_attn = inject_fn(
                    layer=layer,
                    model_config=model_config,
                    comp_config=comp_config_obj,
                    layer_idx=idx,
                    total_layers=total_layers,
                    old_attn=old_attn,
                )
                if new_attn is not None:
                    _attach_new_attention(layer, attn_attr, new_attn, model_type)
                    continue
            except Exception as e:
                logger.warning(
                    "Injector '%s' failed for layer %d: %s. Falling back to builder.",
                    getattr(inject_fn, "__name__", "?"),
                    idx,
                    e,
                )

        if old_attn is None:
            logger.warning("No old attention found in layer %d; skipping.", idx)
            continue

        try:
            hidden_size = _infer_hidden_size(old_attn, attn_attr, model_type)
            num_heads = _infer_num_heads(old_attn, attn_attr, model_type)
            cls_name = _resolve_class_name(inject_fn, model_type)
            new_attn = _build_attention_from_registry(
                cls_name,
                hidden_size=hidden_size,
                num_heads=num_heads,
                config=model_config,
                old_attn=old_attn,
                **safe_params,
            )
            if hasattr(new_attn, "copy_weights"):
                try:
                    new_attn.copy_weights(old_attn)
                except Exception as ex:
                    logger.warning("copy_weights failed: %s", ex)
            p = next(old_attn.parameters(), None)
            if p is not None and isinstance(p.dtype, torch.dtype):
                new_attn.to(p.dtype)
            _attach_new_attention(layer, attn_attr, new_attn, model_type)
        except Exception as e:
            logger.error("Failed to build attention for layer %d: %s", idx, e)
            continue


__all__ = [
    "_build_attention_from_registry",
    "_get_attn_submodule",
    "_attach_new_attention",
    "_replace_layers",
    "_get_valid_mask",
    "_cfg_get",
    "randomized_svd",
    "FactorizedLinear",
    "LoRALinear",
]
