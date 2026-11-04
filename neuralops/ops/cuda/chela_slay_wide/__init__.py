#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
CHELA-SLAY Wide Head Attention CUDA Module.
"""

from __future__ import annotations

from neuralops.ops.cuda.chela_slay_wide.chela_slay_wide_autograd import (
    ChelaSlayWideFunction,
    chela_slay_wide_attention_forward,
)

__all__ = [
    "ChelaSlayWideFunction",
    "chela_slay_wide_attention_forward",
]
