"""Backward-compatibility shim for neuralops.benchmarks.suites.roofline.

This module is deprecated and will be removed in a future release.
Please import from neuralops.benchmarks.suites_v2.roofline instead.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.suites.roofline is deprecated and will be removed in a future release. "
    "Please use neuralops.benchmarks.suites_v2.roofline instead.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks.suites_v2.roofline import *  # noqa: F401, F403
