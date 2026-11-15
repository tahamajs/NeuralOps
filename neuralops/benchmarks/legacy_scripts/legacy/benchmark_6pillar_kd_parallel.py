"""Deprecation forwarding shim for legacy/benchmark_6pillar_kd_parallel."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.legacy.benchmark_6pillar_kd_parallel is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.legacy.benchmark_6pillar_kd_parallel. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.legacy.benchmark_6pillar_kd_parallel import *  # noqa: F403, F401
