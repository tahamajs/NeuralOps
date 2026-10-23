#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Hardware Operator Tier (Tier 1)

Manages native CUDA C++, Triton JIT, and PyTorch reference kernel execution.
"""

from __future__ import annotations

import sys
from neuralops._lazy import _LazyModule

_import_structure = {
    "cuda": ["list_cuda_kernels", "CUDA_KERNEL_CATALOG"],
    "triton": ["triton_dispatch"],
    "pytorch": [],
}

__all__ = ["cuda", "triton", "pytorch"]

module = _LazyModule(
    __name__,
    __file__,
    _import_structure,
    module_spec=__spec__,
)
sys.modules[__name__] = module
