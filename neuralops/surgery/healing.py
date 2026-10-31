#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Post-Surgery Calibration & Subspace Micro-Healing Module.

Provides fast layerwise subspace micro-healing and closed-form Ridge subspace
calibration to eliminate post-transplantation representation shock without full-model
backpropagation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union
import torch
import torch.nn as nn

from neuralops.core._attention_surgery import (
    calibrate_attention_subspace,
    calibrate_surgically_replaced_model,
    heal_replaced_model,
    _heal_replaced_model_impl,
)

__all__ = [
    "heal_replaced_model",
    "calibrate_attention_subspace",
    "calibrate_surgically_replaced_model",
    "_heal_replaced_model_impl",
]
