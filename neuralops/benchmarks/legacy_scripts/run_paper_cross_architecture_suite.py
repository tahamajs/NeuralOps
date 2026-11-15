"""Deprecation forwarding shim for run_paper_cross_architecture_suite."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.run_paper_cross_architecture_suite is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.run_paper_cross_architecture_suite. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.run_paper_cross_architecture_suite import *  # noqa: F403, F401
