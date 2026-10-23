#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Backward compatibility shim for neuralops.compression.quantization."""

from __future__ import annotations

from typing import Any
import neuralops.compression.quantization as _mod

__getattr__ = _mod.__getattr__
__dir__ = _mod.__dir__
__all__ = list(_mod.__all__)
