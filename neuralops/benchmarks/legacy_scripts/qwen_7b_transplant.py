"""Deprecation forwarding shim for qwen_7b_transplant."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.qwen_7b_transplant is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.qwen_7b_transplant. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.qwen_7b_transplant import *  # noqa: F403, F401
