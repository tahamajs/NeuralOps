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
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.kwargs = kwargs.get("kwargs", {})
            self.num_stages = kwargs.get("num_stages", 1)
            self.num_warps = kwargs.get("num_warps", 4)

    class _MockLanguage:
        def __getattr__(self, name: str) -> Any:
            raise ImportError(
                f"Triton language construct 'tl.{name}' is not available because Triton is not installed. "
                "Install neuralops[cuda] on Linux with compatible GPU/drivers to use Triton kernels."
            )

    def _mock_jit(fn: Any = None, *args: Any, **kwargs: Any) -> Any:
        def decorator(f: Any) -> Any:
            def _caller(*call_args: Any, **call_kwargs: Any) -> Any:
                raise ImportError(
                    f"Triton kernel '{getattr(f, '__name__', 'unknown')}' cannot be executed because Triton is not installed. "
                    "Install neuralops[cuda] on Linux with compatible GPU/drivers to use Triton JIT kernels."
                )

            _caller.__name__ = getattr(f, "__name__", "mock_kernel")
            _caller.__doc__ = getattr(f, "__doc__", None)
            _caller.__wrapped__ = f
            return _caller

        if fn is not None and callable(fn):
            return decorator(fn)
        return decorator

    triton = type(
        "triton",
        (),
        {
            "jit": staticmethod(_mock_jit),
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
    FORCE_PYTORCH_FALLBACK,
    FORWARD_ONLY_KERNELS,
    MAX_TRITON_SEQ_LEN,
    SUPPORTED_HEAD_DIMS,
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
