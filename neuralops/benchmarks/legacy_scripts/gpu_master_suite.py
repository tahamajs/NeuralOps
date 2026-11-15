"""Deprecation forwarding shim for gpu_master_suite."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.gpu_master_suite is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.gpu_master_suite. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.gpu_master_suite import *  # noqa: F403, F401
