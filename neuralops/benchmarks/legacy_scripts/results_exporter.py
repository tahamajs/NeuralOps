"""Deprecation forwarding shim for results_exporter."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.results_exporter is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.results_exporter. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.results_exporter import *  # noqa: F403, F401
