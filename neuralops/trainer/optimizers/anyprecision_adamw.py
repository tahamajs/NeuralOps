#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
AnyPrecisionAdamW Optimizer Alias Module.

Provides AnyPrecisionAdamW optimizer and factory functions for reduced precision
momentum/variance training with Kahan summation compensation.
"""

from __future__ import annotations

from neuralops.trainer.optimizers.anyprecision import (
    AnyPrecisionAdamW,
    create_anyprecision_optimizer,
)

__all__ = ["AnyPrecisionAdamW", "create_anyprecision_optimizer"]
