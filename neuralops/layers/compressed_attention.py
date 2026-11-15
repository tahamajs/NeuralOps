#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Compressed Attention layer alias."""

from __future__ import annotations

from neuralops.layers.svd.compressed_attention import (
    CompressedAttention,
    CompressedAttentionLayer,
    CompressedAttentionLayer2,
    inject_compressed_attention,
)

__all__ = [
    "CompressedAttention",
    "CompressedAttentionLayer",
    "CompressedAttentionLayer2",
    "inject_compressed_attention",
]
