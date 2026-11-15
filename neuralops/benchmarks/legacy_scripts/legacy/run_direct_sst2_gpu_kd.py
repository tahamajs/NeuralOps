"""Deprecation forwarding shim for legacy/run_direct_sst2_gpu_kd."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.legacy.run_direct_sst2_gpu_kd is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.legacy.run_direct_sst2_gpu_kd. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.legacy.run_direct_sst2_gpu_kd import *  # noqa: F403, F401
