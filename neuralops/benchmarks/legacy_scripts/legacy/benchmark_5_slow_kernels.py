"""Deprecation forwarding shim for legacy/benchmark_5_slow_kernels."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.legacy.benchmark_5_slow_kernels is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.legacy.benchmark_5_slow_kernels. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.legacy.benchmark_5_slow_kernels import *  # noqa: F403, F401
