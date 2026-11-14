"""Deprecation forwarding shim for benchmark_delta_healing."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.benchmark_delta_healing is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.benchmark_delta_healing. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.benchmark_delta_healing import *  # noqa: F403, F401
