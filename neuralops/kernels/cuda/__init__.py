#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps CUDA Hardware Operators (Lazy Facade over ops.cuda).
"""

from __future__ import annotations

import importlib
from typing import Any

from neuralops.ops.cuda._map import CUDA_KERNEL_CATALOG, _KERNEL_SYMBOL_MAP, _SPECIAL_MAP

CATALOG = CUDA_KERNEL_CATALOG

__all__ = sorted(list(set(CUDA_KERNEL_CATALOG) | set(_KERNEL_SYMBOL_MAP.keys()) | set(_SPECIAL_MAP.keys()) | {"CATALOG"}))


def __getattr__(name: str) -> Any:
    """Resolve CUDA kernels and modules dynamically from neuralops.ops.cuda."""
    if name.startswith("_") and name not in _KERNEL_SYMBOL_MAP:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    import neuralops.ops.cuda as _ops_cuda

    if hasattr(_ops_cuda, name):
        val = getattr(_ops_cuda, name)
        globals()[name] = val
        return val

    if name in CUDA_KERNEL_CATALOG:
        mod = importlib.import_module(f"neuralops.ops.cuda.{name}")
        globals()[name] = mod
        return mod

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    """Execute __dir__ operation."""
    return sorted(list(set(globals().keys()) | set(__all__)))
