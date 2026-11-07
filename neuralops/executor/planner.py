#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Deprecated executor planner module (re-exports from pipeline_transform.planner).
"""

from __future__ import annotations

import warnings

from neuralops.pipeline_transform.planner import plan_recipe

warnings.warn(
    "neuralops.executor.planner is deprecated. Use neuralops.pipeline_transform.planner instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["plan_recipe"]
