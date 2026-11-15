"""Deprecation forwarding shim for benchmark_config."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.benchmark_config is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.benchmark_config. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.benchmark_config import *  # noqa: F403, F401
