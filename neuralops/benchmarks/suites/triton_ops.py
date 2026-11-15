"""Backward-compatibility shim for neuralops.benchmarks.suites.triton_ops.

This module is deprecated and will be removed in a future release.
Please import from neuralops.benchmarks.suites_v2.triton_ops instead.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.suites.triton_ops is deprecated and will be removed in a future release. "
    "Please use neuralops.benchmarks.suites_v2.triton_ops instead.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks.suites_v2.triton_ops import *  # noqa: F401, F403
