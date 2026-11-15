"""Backward-compatibility shim for neuralops.benchmarks.suites.attention.

This module is deprecated and will be removed in a future release.
Please import from neuralops.benchmarks.suites_v2.attention instead.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.suites.attention is deprecated and will be removed in a future release. "
    "Please use neuralops.benchmarks.suites_v2.attention instead.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks.suites_v2.attention import *  # noqa: F401, F403
