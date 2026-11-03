#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Backward compatibility shim for neuralops.ops.triton."""

from __future__ import annotations

from typing import Any
import sys


HAS_TRITON = True


def __getattr__(name: str) -> Any:
    if name == "HAS_TRITON":
        return True
    if name == "fast_linear_attention":
        from neuralops.ops.triton.linear_attn import linear_attn
        return linear_attn
    import neuralops.ops.triton as _ops_triton
    return getattr(_ops_triton, name)


def __dir__() -> list[str]:
    import neuralops.ops.triton as _ops_triton
    return list(_ops_triton.__dir__()) + ["HAS_TRITON", "fast_linear_attention"]
