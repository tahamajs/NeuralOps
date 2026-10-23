#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Native CUDA C++ Kernel Operators (Tier 1)

Dispatches compiled CUDA extension operations and PyTorch reference fallbacks.
"""

from __future__ import annotations

from typing import Any
import neuralops.core.cuda_kernels as _nck

CUDA_KERNEL_CATALOG = _nck.CUDA_KERNEL_CATALOG
list_cuda_kernels = _nck.list_cuda_kernels


def __getattr__(name: str) -> Any:
    return getattr(_nck, name)


def __dir__() -> list[str]:
    return list(_nck.__all__)


__all__ = list(_nck.__all__)
