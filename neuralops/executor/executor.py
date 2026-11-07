#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Deprecated recipe execution engine module.

DEPRECATION NOTICE:
This module is deprecated in favor of `neuralops.pipeline_transform.executor`
and will be removed in version 2.2.0.
"""

from __future__ import annotations

import warnings

from neuralops.pipeline_transform.executor import (
    execute_plan as canonical_execute_plan,
    execute_recipe as canonical_execute_recipe,
)

warnings.warn(
    "neuralops.executor.executor is deprecated and scheduled for removal in NeuralOps 2.2.0. "
    "Use neuralops.pipeline_transform.executor instead.",
    DeprecationWarning,
    stacklevel=2,
)

execute_plan = canonical_execute_plan
execute_recipe = canonical_execute_recipe

__all__ = ["execute_plan", "execute_recipe"]
