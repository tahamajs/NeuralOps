"""Deprecation forwarding shim for run_large_model_7b_bench."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.run_large_model_7b_bench is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.run_large_model_7b_bench. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.run_large_model_7b_bench import *  # noqa: F403, F401
