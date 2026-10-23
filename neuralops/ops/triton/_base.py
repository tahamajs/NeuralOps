#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Base declarations and constants for Triton kernels.
Zero imports from high-level registries or kernels.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("TritonKernels")

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

    class _MockConfig:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs.get("kwargs", {})
            self.num_stages = kwargs.get("num_stages", 1)
            self.num_warps = kwargs.get("num_warps", 4)

    class _MockLanguage:
        def __getattr__(self, name: str) -> Any:
            return lambda *args, **kwargs: None

    triton = type(
        "triton",
        (),
        {
            "jit": lambda f: f,
            "heuristics": lambda *a, **kw: lambda f: f,
            "autotune": lambda *a, **kw: lambda f: f,
            "Config": _MockConfig,
            "cdiv": lambda a, b: (a + b - 1) // b,
            "next_power_of_2": lambda n: 1 << (n - 1).bit_length() if n > 0 else 1,
        },
    )
    tl = _MockLanguage()

from neuralops.ops.triton._types import (
    BLOCK_SIZE,
    MAX_TRITON_SEQ_LEN,
    SUPPORTED_HEAD_DIMS,
    FORWARD_ONLY_KERNELS,
    FORCE_PYTORCH_FALLBACK,
    check_triton_suitability,
    log_fallback_reason,
)

__all__ = [
    "HAS_TRITON",
    "triton",
    "tl",
    "BLOCK_SIZE",
    "MAX_TRITON_SEQ_LEN",
    "SUPPORTED_HEAD_DIMS",
    "FORWARD_ONLY_KERNELS",
    "FORCE_PYTORCH_FALLBACK",
    "check_triton_suitability",
    "log_fallback_reason",
]
