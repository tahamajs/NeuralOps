#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Native Sparse Attention (NSA) canonical layer export."""

from __future__ import annotations

from neuralops.layers.sparse.nsa_attention import (
    LearnedBlockRouter,
    NativeSparseAttention,
    inject_nsa,
)

__all__ = ["NativeSparseAttention", "LearnedBlockRouter", "inject_nsa"]
