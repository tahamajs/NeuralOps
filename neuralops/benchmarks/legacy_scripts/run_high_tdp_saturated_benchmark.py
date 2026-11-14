"""Deprecation forwarding shim for run_high_tdp_saturated_benchmark."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.run_high_tdp_saturated_benchmark is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.run_high_tdp_saturated_benchmark. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.run_high_tdp_saturated_benchmark import *  # noqa: F403, F401
