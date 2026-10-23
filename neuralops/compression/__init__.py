#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Unified Compression Engines (Tier 2).
"""

from __future__ import annotations

import importlib
from typing import Any

from neuralops.core.base import BaseCompressor
from neuralops.core.compression_pipeline import compress_model
from neuralops.core.registry import COMPRESSOR_REGISTRY, get_compressor, list_compressors

COMPRESSOR_REGISTRY = COMPRESSOR_REGISTRY
get_compressor = get_compressor
list_compressors = list_compressors
BaseCompressor = BaseCompressor
compress_model = compress_model

_SUBPACKAGES = [
    "experimental",
    "kv_cache",
    "low_rank",
    "pruning",
    "quantization",
    "svd",
    "tensor_decomp",
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
    names = (
        set(globals())
        | set(_SUBPACKAGES)
        | {"COMPRESSOR_REGISTRY", "BaseCompressor", "compress_model", "get_compressor", "list_compressors"}
    )
    for subpackage in _SUBPACKAGES:
        try:
            mod = importlib.import_module(f".{subpackage}", __package__)
            names.update(getattr(mod, "__all__", []))
        except Exception:
            pass
    return sorted(names)


__all__ = sorted(list(set(__dir__())))
