#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Autograd Core Modules.
"""

from __future__ import annotations

from neuralops.core.autograd.flash_attention import FlashAttentionFunction
from neuralops.core.autograd.hvp import compute_hvp, verify_autograd_graph_cleanliness

__all__ = [
    "FlashAttentionFunction",
    "compute_hvp",
    "verify_autograd_graph_cleanliness",
]

from neuralops.core.autograd import flash_attention, fused_backward, hvp
