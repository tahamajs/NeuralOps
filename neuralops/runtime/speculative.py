#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Runtime Speculative Decoding Module.

Canonical alias and entrypoint for SelfSpeculativeDecoder in runtime pipelines.
"""

from __future__ import annotations

from neuralops.inference.speculative_decoding import (
    SelfSpeculativeDecoder,
    SpeculativeVerificationResult,
)

__all__ = ["SelfSpeculativeDecoder", "SpeculativeVerificationResult"]
