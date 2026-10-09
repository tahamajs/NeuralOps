"""Backward-compatible import forwarding for neuralops.benchmarks.result."""

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
