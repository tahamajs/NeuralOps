#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Mixture of Block Attention (MoBA) layer alias."""

from __future__ import annotations

from neuralops.layers.sparse.moba_attention import (
    MoBAAttention,
    inject_moba,
)

__all__ = ["MoBAAttention", "inject_moba"]
