"""Deprecation forwarding shim for gpu_telemetry."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.gpu_telemetry is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.gpu_telemetry. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.gpu_telemetry import *  # noqa: F403, F401
