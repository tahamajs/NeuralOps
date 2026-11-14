"""Deprecation forwarding shim for gpu_stress_heavy_trainer."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.gpu_stress_heavy_trainer is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.gpu_stress_heavy_trainer. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.gpu_stress_heavy_trainer import *  # noqa: F403, F401
