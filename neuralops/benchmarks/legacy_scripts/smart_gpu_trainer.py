"""Deprecation forwarding shim for smart_gpu_trainer."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.smart_gpu_trainer is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.smart_gpu_trainer. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.smart_gpu_trainer import *  # noqa: F403, F401
