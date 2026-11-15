#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Native Sparse Attention (NSA) layer."""

from __future__ import annotations

from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.layers.sparse.nsa_attention import (
    LearnedBlockRouter,
    NativeSparseAttention as _BaseNSA,
    inject_nsa,
)


@ATTENTION_REGISTRY.register(
    "nsa",
    is_alias=True,
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class NSAAttention(_BaseNSA):
    """Canonical NSA Attention alias."""


NativeSparseAttention = _BaseNSA
__all__ = ["NSAAttention", "NativeSparseAttention", "LearnedBlockRouter", "inject_nsa"]
