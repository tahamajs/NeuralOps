#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Attention Layers Tier (Tier 2)

Houses all subquadratic attention families (linear, svd, ssm, gated, hybrid, sparse, adaptive, modern, fast_kernels, dense, experimental).
"""

from __future__ import annotations

import importlib
from typing import Any

from neuralops._optional import _require_torch
_require_torch("neuralops.layers")

from neuralops.core.registry import ATTENTION_REGISTRY, get_attention, list_attentions

ATTENTION_REGISTRY = ATTENTION_REGISTRY
get_attention = get_attention
list_attentions = list_attentions

_SUBPACKAGES = [
    "adaptive",
    "dense",
    "experimental",
    "fast_kernels",
    "gated",
    "hybrid",
    "linear",
    "modern",
    "sparse",
    "ssm",
    "svd",
]


def __getattr__(name: str) -> Any:
    if name.startswith("_"):
        raise AttributeError(f"module '{__name__}' has no attribute {name!r}")
    if name in _SUBPACKAGES:
        module = importlib.import_module(f".{name}", __package__)
        globals()[name] = module
        return module
    for subpackage in _SUBPACKAGES:
        try:
            module = importlib.import_module(f".{subpackage}", __package__)
            if hasattr(module, name):
                val = getattr(module, name)
                globals()[name] = val
                return val
        except (ImportError, AttributeError):
            continue
    raise AttributeError(f"module '{__name__}' has no attribute {name!r}")


def __dir__() -> list[str]:
    names = set(globals()) | set(_SUBPACKAGES) | {"ATTENTION_REGISTRY", "get_attention", "list_attentions"}
    for subpackage in _SUBPACKAGES:
        try:
            mod = importlib.import_module(f".{subpackage}", __package__)
            names.update(getattr(mod, "__all__", []))
        except Exception:
            pass
    return sorted(names)


__all__ = sorted(list(set(__dir__())))
