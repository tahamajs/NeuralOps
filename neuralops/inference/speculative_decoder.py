#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Speculative Decoder Module

Canonical entrypoint and compatibility alias for SelfSpeculativeDecoder,
SpeculativeDecodeStats, and speculative decoding verification routines.
"""

from __future__ import annotations

from neuralops.inference.speculative_decoding import SelfSpeculativeDecoder, SpeculativeDecodeStats, gamma_sweep_harness

# Canonical alias
SpeculativeDecoder = SelfSpeculativeDecoder

__all__ = [
    "SelfSpeculativeDecoder",
    "SpeculativeDecoder",
    "SpeculativeDecodeStats",
    "gamma_sweep_harness",
]
