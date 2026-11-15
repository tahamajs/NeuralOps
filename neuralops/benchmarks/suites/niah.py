"""Backward-compatibility shim for neuralops.benchmarks.suites.niah.

This module is deprecated and will be removed in a future release.
Please import from neuralops.benchmarks.suites_v2.niah instead.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.suites.niah is deprecated and will be removed in a future release. "
    "Please use neuralops.benchmarks.suites_v2.niah instead.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks.suites_v2.niah import *  # noqa: F401, F403
