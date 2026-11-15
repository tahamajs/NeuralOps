"""Deprecation forwarding shim for gpu_master_suite."""

from __future__ import annotations

import sys
import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.gpu_master_suite is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.gpu_master_suite. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

import neuralops.benchmarks._legacy.gpu_master_suite as _impl

# Forward everything and alias module so monkeypatching reflects directly on implementation
globals().update({k: getattr(_impl, k) for k in dir(_impl)})
sys.modules[__name__] = _impl
