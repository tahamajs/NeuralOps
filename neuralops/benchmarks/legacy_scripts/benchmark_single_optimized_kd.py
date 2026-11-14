"""Deprecation forwarding shim for benchmark_single_optimized_kd."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.benchmark_single_optimized_kd is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.benchmark_single_optimized_kd. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.benchmark_single_optimized_kd import *  # noqa: F403, F401
