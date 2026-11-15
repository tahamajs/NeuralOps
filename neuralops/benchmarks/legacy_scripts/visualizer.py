"""Deprecation forwarding shim for visualizer."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.visualizer is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.visualizer. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.visualizer import *  # noqa: F403, F401
