#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Attention Surgery and Layer Replacement Infrastructure (Legacy Shim).

This module provides backwards compatibility for legacy attention replacement call sites.
New implementations should directly utilize neuralops.surgery.director.UniversalSurgeryDirector.
"""

from __future__ import annotations

import logging
from typing import Any

import torch.nn as nn

from neuralops.models.surgery import (
    FUSED_QKV_REGEX,
    PROJ_REGEX_PATTERNS,
    _copy_projection_weights,
    _find_submodule_by_regex,
    _heal_replaced_model_impl,
    _slice_fused_qkv_weights,
    _svd_factorize_weights,
    adapt_decoder_attention_forward,
    calibrate_attention_subspace,
    calibrate_surgically_replaced_model,
    compute_reconstruction_error,
    heal_replaced_model,
    slice_fused_qkv,
    split_and_set_qkv_weights,
)
from neuralops.core.replace_attention import replace_attention_in_model
from neuralops.surgery.director import UniversalSurgeryDirector
from neuralops.surgery.layer_replacement import replace_attention_recursive

logger = logging.getLogger("neuralops.surgery.attention_replace")


def apply_backbone_surgery_hooks(
    model: nn.Module,
    model_type: str = "llama",
    attach_forward_adapter: bool = True,
) -> list[str]:
    """Hooks RoPE & GQA backbone decoders (LLaMA, Mistral, Qwen2, Gemma) to ensure

    injected modules handle position_embeddings, position_ids, and return tuples safely.

    Attaches forward adapters to handle keyword arguments (e.g. position_embeddings,
    cache_position, past_key_value) and guarantees that the module's return value
    is always the standard 3-tuple expected by transformer decoder layers:
    (attn_output, attn_weights, past_key_value).

    Args:
        model: Target model or container module.
        model_type: Backbone architecture family name (e.g. 'llama', 'mistral', 'qwen2', 'gemma').
        attach_forward_adapter: Whether to wrap attention module execution with runtime
            argument adaptation and 3-tuple enforcement. Defaults to True.

    Returns:
        List of hooked module names.
    """
    mtype = model_type.lower()
    supported = ("llama", "mistral", "qwen", "qwen2", "qwen2_5", "qwen3_5", "gemma", "gemma2")
    if not any(sup in mtype for sup in supported):
        return []

    hooked_names: list[str] = []
    for name, module in model.named_modules():
        leaf = name.split(".")[-1].lower()
        if leaf in ("self_attn", "attention", "attn"):
            if not hasattr(module, "supports_gqa"):
                module.supports_gqa = True
            if not hasattr(module, "supports_rope"):
                module.supports_rope = True

            if attach_forward_adapter and not getattr(module, "_backbone_surgery_hooks_applied", False):
                module._backbone_surgery_hooks_applied = True
                orig_forward = module.forward

                def _make_hooked_forward(fn: Any, mod: nn.Module) -> Any:
                    import inspect

                    try:
                        sig = inspect.signature(fn)
                        accepted_params = set(sig.parameters.keys())
                        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                    except (ValueError, TypeError):
                        accepted_params = set()
                        has_var_kw = True

                    def _adapted_forward(*args: Any, **kwargs: Any) -> Any:
                        call_kwargs = dict(kwargs)
                        output_attentions = bool(call_kwargs.get("output_attentions", False))
                        if not has_var_kw and accepted_params:
                            call_kwargs = {k: v for k, v in call_kwargs.items() if k in accepted_params}

                        out = None
                        try:
                            out = fn(*args, **call_kwargs)
                        except TypeError:
                            if args:
                                h = args[0]
                                m = kwargs.get("attention_mask", None)
                                try:
                                    out = fn(h, attention_mask=m) if m is not None else fn(h)
                                except TypeError:
                                    out = fn(h)
                            elif "hidden_states" in kwargs:
                                h = kwargs["hidden_states"]
                                m = kwargs.get("attention_mask", None)
                                try:
                                    out = fn(h, attention_mask=m) if m is not None else fn(h)
                                except TypeError:
                                    out = fn(h)
                            else:
                                raise

                        if isinstance(out, tuple):
                            if len(out) >= 3:
                                return out
                            elif len(out) == 2:
                                attn_w = out[1] if output_attentions else None
                                past_kv = out[1] if not output_attentions else None
                                return (out[0], attn_w, past_kv)
                            elif len(out) == 1:
                                return (out[0], None, None)
                        return (out, None, None)

                    return _adapted_forward

                module.forward = _make_hooked_forward(orig_forward, module)
            hooked_names.append(name)

    return hooked_names


def replace_attention(*args: Any, **kwargs: Any) -> Any:
    """Drop-in replace attention layers in any PyTorch / HuggingFace model."""
    from neuralops.api import replace_attention as _ra

    return _ra(*args, **kwargs)

__all__ = [
    "PROJ_REGEX_PATTERNS",
    "FUSED_QKV_REGEX",
    "_find_submodule_by_regex",
    "compute_reconstruction_error",
    "_slice_fused_qkv_weights",
    "slice_fused_qkv",
    "_copy_projection_weights",
    "split_and_set_qkv_weights",
    "_svd_factorize_weights",
    "adapt_decoder_attention_forward",
    "apply_backbone_surgery_hooks",
    "calibrate_surgically_replaced_model",
    "calibrate_attention_subspace",
    "heal_replaced_model",
    "_heal_replaced_model_impl",
    "replace_attention_in_model",
    "replace_attention_recursive",
    "replace_attention",
    "UniversalSurgeryDirector",
]
