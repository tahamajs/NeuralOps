#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Compatibility facade; kernel implementations live under core.triton_kernels."""

from __future__ import annotations

from typing import Any

from neuralops.core import triton_kernels as _core

__all__ = list(_core.__all__)


def __getattr__(name: str) -> Any:
    """Resolve legacy exports through the canonical core Triton package."""
    try:
        return getattr(_core, name)
    except AttributeError:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from None


def __dir__() -> list[str]:
    """List exports exposed by the compatibility facade."""
    return sorted(set(globals()) | set(__all__))
