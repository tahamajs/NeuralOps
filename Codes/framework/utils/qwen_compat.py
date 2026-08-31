"""Compatibility helpers for decoder-only Qwen attention blocks.

This module only exposes the names expected by legacy attention/compressor
implementations.  It does not replace Qwen's rotary or causal attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class QwenConfigProxy:
    """Small, read-through config view with the common attention fields."""

    source: Any
    hidden_size: int
    num_attention_heads: int
    head_dim: int
    num_hidden_layers: int

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source, name)


def is_qwen_model(model: nn.Module) -> bool:
    config = getattr(model, "config", None)
    return getattr(config, "model_type", "") in {"qwen2", "qwen2.5"} or "qwen" in model.__class__.__name__.lower()


def qwen_config_proxy(model_or_config: Any) -> QwenConfigProxy:
    config = getattr(model_or_config, "config", model_or_config)
    hidden = int(getattr(config, "hidden_size"))
    heads = int(getattr(config, "num_attention_heads"))
    return QwenConfigProxy(
        source=config,
        hidden_size=hidden,
        num_attention_heads=heads,
        head_dim=int(getattr(config, "head_dim", hidden // heads)),
        num_hidden_layers=int(getattr(config, "num_hidden_layers", 0)),
    )


def install_qwen_aliases(model: nn.Module) -> nn.Module:
    """Install non-structural projection aliases on every Qwen attention block."""
    if not is_qwen_model(model):
        return model
    for module in model.modules():
        if not hasattr(module, "q_proj"):
            continue
        # object.__setattr__ avoids registering duplicate parameter paths in
        # state_dict while still making getattr(module, "q_lin") work.
        for legacy, native in (("q_lin", "q_proj"), ("k_lin", "k_proj"),
                               ("v_lin", "v_proj"), ("out_lin", "o_proj")):
            if not hasattr(module, legacy) and hasattr(module, native):
                object.__setattr__(module, legacy, getattr(module, native))
    return model


def is_dtype_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, (RuntimeError, NotImplementedError)) and any(token in text for token in (
        "half", "float16", "geqrf", "dtype", "linalg", "qr decomposition", "not implemented",
    ))


def is_architecture_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(token in text for token in (
        "shape", "size mismatch", "unexpected keyword", "unexpected argument", "has no attribute",
        "not enough values to unpack", "attention mask", "position_embeddings",
    ))
