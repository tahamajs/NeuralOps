#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""FlashAttention-2 Layer.

Optimized FlashAttention-2 implementation (Dao, 2023) improving work partitioning
across thread blocks, reducing non-matmul FLOPs, and enhancing causal decoding throughput.
"""

from __future__ import annotations

from typing import Any

from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.layers.flash_attention import FlashAttention

__all__ = [
    "FlashAttention2",
    "flash_attention_2",
]


@ATTENTION_REGISTRY.register(
    "flash_attention_2",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class FlashAttention2(FlashAttention):
    """FlashAttention-2 neural operator."""

    FAMILY: str = "fast_kernels"


flash_attention_2 = FlashAttention2
