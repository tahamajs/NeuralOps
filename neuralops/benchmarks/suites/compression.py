"""Backward-compatibility shim for neuralops.benchmarks.suites.compression.

This module is deprecated and will be removed in a future release.
Please import from neuralops.benchmarks.suites_v2.compression instead.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.suites.compression is deprecated and will be removed in a future release. "
    "Please use neuralops.benchmarks.suites_v2.compression instead.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks.suites_v2.compression import *  # noqa: F401, F403
