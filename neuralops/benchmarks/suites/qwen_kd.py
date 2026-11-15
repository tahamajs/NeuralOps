"""Backward-compatibility shim for neuralops.benchmarks.suites.qwen_kd.

This module is deprecated and will be removed in a future release.
Please import from neuralops.benchmarks.suites_v2.qwen_kd instead.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.suites.qwen_kd is deprecated and will be removed in a future release. "
    "Please use neuralops.benchmarks.suites_v2.qwen_kd instead.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks.suites_v2.qwen_kd import *  # noqa: F401, F403
