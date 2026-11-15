#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Hamming Distance Sparse Attention layer alias."""

from __future__ import annotations

from neuralops.layers.sparse.hamming_attention import (
    HammingAttention,
    inject_hamming_attention,
)

inject_hamming = inject_hamming_attention

__all__ = ["HammingAttention", "inject_hamming", "inject_hamming_attention"]
