"""Deprecation forwarding shim for legacy/benchmark_10_phases_full_coverage."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.legacy.benchmark_10_phases_full_coverage is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.legacy.benchmark_10_phases_full_coverage. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.legacy.benchmark_10_phases_full_coverage import *  # noqa: F403, F401
