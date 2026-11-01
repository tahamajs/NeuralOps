#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Transformation Executor and Planner Subsystem.
"""

from neuralops.executor.executor import execute_plan, execute_recipe
from neuralops.executor.layer_parallel import LayerParallelExecutor
from neuralops.executor.plan import ExecutionPlan, Stage
from neuralops.executor.planner import plan_recipe
from neuralops.executor.result import PipelineResult, StageRecord
from neuralops.executor.snapshot import ModelSnapshot

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
