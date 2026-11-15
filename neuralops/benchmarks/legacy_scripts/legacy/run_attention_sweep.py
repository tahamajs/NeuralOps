"""Deprecation forwarding shim for legacy/run_attention_sweep."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.legacy.run_attention_sweep is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.legacy.run_attention_sweep. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.legacy.run_attention_sweep import *  # noqa: F403, F401
