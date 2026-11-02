#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Backward-compatibility shim for neuralops.ops.triton.sliding_window."""

from __future__ import annotations

from neuralops.ops.triton.sliding_window import (
    sliding_window_attention,
    sliding_window_attention_kernel,
    sliding_window_pytorch,
    sliding_window_triton,
)

__all__ = [
    "sliding_window_attention",
    "sliding_window_attention_kernel",
    "sliding_window_pytorch",
    "sliding_window_triton",
]
