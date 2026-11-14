"""Deprecation forwarding shim for benchmark_10_phases_speedup_vs_pytorch."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.benchmark_10_phases_speedup_vs_pytorch is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.benchmark_10_phases_speedup_vs_pytorch. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.benchmark_10_phases_speedup_vs_pytorch import *  # noqa: F403, F401
