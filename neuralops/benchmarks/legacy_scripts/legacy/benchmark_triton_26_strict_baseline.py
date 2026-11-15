"""Deprecation forwarding shim for legacy/benchmark_triton_26_strict_baseline."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.legacy.benchmark_triton_26_strict_baseline is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.legacy.benchmark_triton_26_strict_baseline. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.legacy.benchmark_triton_26_strict_baseline import *  # noqa: F403, F401
