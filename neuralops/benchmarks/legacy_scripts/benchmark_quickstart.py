"""Deprecation forwarding shim for benchmark_quickstart."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.benchmark_quickstart is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.benchmark_quickstart. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.benchmark_quickstart import *  # noqa: F403, F401
