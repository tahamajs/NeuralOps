#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Deprecated executor plan module (re-exports from pipeline_transform.plan).
"""

from __future__ import annotations

import warnings

from neuralops.pipeline_transform.plan import ExecutionPlan, Stage

warnings.warn(
    "neuralops.executor.plan is deprecated. Use neuralops.pipeline_transform.plan instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ExecutionPlan", "Stage"]
