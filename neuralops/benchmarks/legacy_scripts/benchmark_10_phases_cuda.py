"""Deprecation forwarding shim for benchmark_10_phases_cuda."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.benchmark_10_phases_cuda is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.benchmark_10_phases_cuda. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.benchmark_10_phases_cuda import *  # noqa: F403, F401
