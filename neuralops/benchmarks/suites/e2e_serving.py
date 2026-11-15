"""Backward-compatibility shim for neuralops.benchmarks.suites.e2e_serving.

This module is deprecated and will be removed in a future release.
Please import from neuralops.benchmarks.suites_v2.e2e_serving instead.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.suites.e2e_serving is deprecated and will be removed in a future release. "
    "Please use neuralops.benchmarks.suites_v2.e2e_serving instead.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks.suites_v2.e2e_serving import *  # noqa: F401, F403
