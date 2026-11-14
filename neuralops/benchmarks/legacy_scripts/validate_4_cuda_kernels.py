"""Deprecation forwarding shim for validate_4_cuda_kernels."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.validate_4_cuda_kernels is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.validate_4_cuda_kernels. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.validate_4_cuda_kernels import *  # noqa: F403, F401
