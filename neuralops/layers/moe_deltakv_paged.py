#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Compatibility alias for neuralops.layers.sparse.moe_deltakv_paged."""

from neuralops.layers.sparse.moe_deltakv_paged import (
    MoEDeltaKVPagedAttention,
    low_rank_delta_compress,
    ste_quantize,
)

__all__ = ["MoEDeltaKVPagedAttention", "ste_quantize", "low_rank_delta_compress"]
