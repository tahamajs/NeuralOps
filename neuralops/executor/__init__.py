#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Deprecated execution stack facade.

DEPRECATION NOTICE:
`neuralops.executor` is deprecated in favor of `neuralops.pipeline_transform`
and is scheduled for removal in NeuralOps 2.2.0. Please import from
`neuralops.pipeline_transform` instead.
"""

from __future__ import annotations

import warnings

from neuralops.executor.executor import execute_recipe
from neuralops.executor.layer_parallel import LayerParallelExecutor
from neuralops.pipeline_transform.executor import execute_plan
from neuralops.pipeline_transform.plan import ExecutionPlan, Stage
from neuralops.pipeline_transform.planner import plan_recipe
from neuralops.pipeline_transform.result import PipelineResult, StageRecord
from neuralops.pipeline_transform.snapshot import Snapshot as ModelSnapshot

warnings.warn(
    "neuralops.executor is deprecated. Use neuralops.pipeline_transform instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "ExecutionPlan",
    "LayerParallelExecutor",
    "ModelSnapshot",
    "PipelineResult",
    "Stage",
    "StageRecord",
    "execute_plan",
    "execute_recipe",
    "plan_recipe",
]
