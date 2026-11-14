"""Deprecation forwarding shim for benchmark_logger."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.benchmark_logger is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.benchmark_logger. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.benchmark_logger import *  # noqa: F403, F401
