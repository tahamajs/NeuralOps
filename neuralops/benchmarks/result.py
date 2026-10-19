#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Backward-compatible import forwarding for neuralops.benchmarks.result

Implements result functionality for NeuralOps.
"""

from neuralops.benchmarks.results.schema import (
    ArtifactPaths,
    BaselineComparison,
    BenchmarkResult,
    DatasetMetadata,
    EnvironmentMetadata,
    HardwareUsage,
    ModelMetadata,
    ModuleMetadata,
    PushInfo,
    ReproducibilityInfo,
    RunMetadata,
)

__all__ = [
    "ModuleMetadata",
    "ModelMetadata",
    "DatasetMetadata",
    "BenchmarkResult",
    "RunMetadata",
    "EnvironmentMetadata",
    "HardwareUsage",
    "BaselineComparison",
    "ReproducibilityInfo",
    "ArtifactPaths",
    "PushInfo",
]
