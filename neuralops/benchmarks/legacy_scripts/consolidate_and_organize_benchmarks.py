"""Deprecation forwarding shim for consolidate_and_organize_benchmarks."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.consolidate_and_organize_benchmarks is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.consolidate_and_organize_benchmarks. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.consolidate_and_organize_benchmarks import *  # noqa: F403, F401
