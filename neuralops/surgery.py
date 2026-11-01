#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Neural Architecture Surgery Facade & ModelSurgery API.
"""

from __future__ import annotations

from neuralops.surgery.model_surgery import ModelSurgery
from neuralops.surgery.surgery import *  # noqa: F403
from neuralops.surgery.surgery import __all__ as _surgery_all

__all__ = ["ModelSurgery", *_surgery_all]
