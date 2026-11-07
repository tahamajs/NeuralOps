#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Compatibility bridge for neuralops.layers.attention -> neuralops.layers.
"""

from __future__ import annotations

import sys
from typing import Any

import neuralops.layers as _layers

sys.modules["neuralops.layers.attention"] = _layers

__all__ = getattr(_layers, "__all__", [])


def __getattr__(name: str) -> Any:
    return getattr(_layers, name)
