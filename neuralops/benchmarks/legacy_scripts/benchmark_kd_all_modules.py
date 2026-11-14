"""Deprecation forwarding shim for benchmark_kd_all_modules."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.benchmark_kd_all_modules is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.benchmark_kd_all_modules. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.benchmark_kd_all_modules import *  # noqa: F403, F401
