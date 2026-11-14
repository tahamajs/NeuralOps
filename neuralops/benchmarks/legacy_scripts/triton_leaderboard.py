"""Deprecation forwarding shim for triton_leaderboard."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.triton_leaderboard is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.triton_leaderboard. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.triton_leaderboard import *  # noqa: F403, F401
