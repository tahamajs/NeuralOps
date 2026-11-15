"""Deprecation forwarding shim for legacy/benchmark_long_context_phase12."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.legacy.benchmark_long_context_phase12 is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.legacy.benchmark_long_context_phase12. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.legacy.benchmark_long_context_phase12 import *  # noqa: F403, F401
