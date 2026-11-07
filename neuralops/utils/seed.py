#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Random Seed Utilities

Sets deterministic random seeds across PyTorch, NumPy, and Python standard libraries.
"""

from __future__ import annotations

from neuralops.core.utils.general import set_seed
from neuralops.core.utils.general import set_seed as set_all_seeds

__all__ = ["set_all_seeds", "set_seed"]
