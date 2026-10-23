#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Hardware and Dependency Guards

Guards for optional heavy runtimes (PyTorch, Triton, CUDA) ensuring light imports.
"""

from __future__ import annotations

import sys
from typing import Optional

from neuralops.core.errors import TorchNotInstalledError, TritonNotInstalledError

_TORCH_AVAILABLE: Optional[bool] = None
_TRITON_AVAILABLE: Optional[bool] = None


def _torch_available() -> bool:
    global _TORCH_AVAILABLE
    if _TORCH_AVAILABLE is None:
        if "torch" in sys.modules:
            _TORCH_AVAILABLE = sys.modules["torch"] is not None
        else:
            try:
                import torch  # noqa: F401
                _TORCH_AVAILABLE = True
            except (ImportError, ModuleNotFoundError):
                _TORCH_AVAILABLE = False
    return _TORCH_AVAILABLE


def _require_torch(feature: str = "this operation") -> None:
    if not _torch_available():
        raise TorchNotInstalledError(
            f"PyTorch is required for {feature}. "
            f"Install with: pip install 'neuralops[cpu]' "
            f"or 'neuralops[cuda]' for GPU support."
        )


def _triton_available() -> bool:
    global _TRITON_AVAILABLE
    if _TRITON_AVAILABLE is None:
        if "triton" in sys.modules:
            _TRITON_AVAILABLE = sys.modules["triton"] is not None
        else:
            try:
                import triton  # noqa: F401
                _TRITON_AVAILABLE = True
            except (ImportError, ModuleNotFoundError):
                _TRITON_AVAILABLE = False
    return _TRITON_AVAILABLE


def _require_triton(feature: str = "this operation") -> None:
    _require_torch(feature)
    if not _triton_available():
        raise TritonNotInstalledError(
            f"Triton is required for {feature}. "
            f"Install with: pip install 'neuralops[cuda]' "
            f"(Linux only; use CPU fallback otherwise)."
        )
