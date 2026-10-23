#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Core CUDA Kernels Hardware Operators (Backward-Compatibility Shim).
"""

from __future__ import annotations

from typing import Any
import neuralops.ops.cuda as _ops_cuda
from neuralops.core.cuda_kernels._map import (
    CUDA_KERNEL_CATALOG,
    _KERNEL_SYMBOL_MAP,
    _SPECIAL_MAP,
)
from neuralops.ops.cuda import list_cuda_kernels, __all__

CUDA_KERNEL_CATALOG = CUDA_KERNEL_CATALOG
list_cuda_kernels = list_cuda_kernels
__all__ = list(__all__)


def __getattr__(name: str) -> Any:
    return getattr(_ops_cuda, name)


def __dir__() -> list[str]:
    return list(_ops_cuda.__dir__())


if "annotations" in globals():
    del annotations
