"""Deprecation forwarding shim for benchmark_all_triton_comprehensive."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.benchmark_all_triton_comprehensive is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.benchmark_all_triton_comprehensive. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.benchmark_all_triton_comprehensive import *  # noqa: F403, F401
