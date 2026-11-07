#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Transactional Model Transformation Pipeline, Planning, and Execution.
"""

from __future__ import annotations

from neuralops.pipeline_transform.auto_config import suggest_config
from neuralops.pipeline_transform.dependencies import (
    Conflict,
    ConflictDetector,
    ConflictSeverity,
    DependencyGraph,
    DependencyNode,
    analyze_recipe,
)
from neuralops.pipeline_transform.errors import ConfigurationError, PlanningError
from neuralops.pipeline_transform.executor import execute_plan, execute_recipe
from neuralops.pipeline_transform.hardware_planner import (
    KNOWN_HARDWARE_PROFILES,
    HardwareProfile,
    ResourceEstimate,
    estimate_resources,
)
from neuralops.pipeline_transform.plan import ExecutionPlan, Stage
from neuralops.pipeline_transform.planner import plan_recipe
from neuralops.pipeline_transform.result import PipelineError, PipelineResult, StageRecord
from neuralops.pipeline_transform.rollback import RollbackError
from neuralops.pipeline_transform.snapshot import ModelSnapshot, Snapshot, capture, restore

__all__ = [
    "Conflict",
    "ConflictDetector",
    "ConflictSeverity",
    "ConfigurationError",
    "DependencyGraph",
    "DependencyNode",
    "ExecutionPlan",
    "HardwareProfile",
    "KNOWN_HARDWARE_PROFILES",
    "ModelSnapshot",
    "PipelineError",
    "PipelineResult",
    "PlanningError",
    "ResourceEstimate",
    "RollbackError",
    "Snapshot",
    "Stage",
    "StageRecord",
    "analyze_recipe",
    "capture",
    "estimate_resources",
    "execute_plan",
    "execute_recipe",
    "plan_recipe",
    "restore",
    "suggest_config",
]
