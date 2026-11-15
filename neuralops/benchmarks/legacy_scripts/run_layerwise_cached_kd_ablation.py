"""Deprecation forwarding shim for run_layerwise_cached_kd_ablation."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.run_layerwise_cached_kd_ablation is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.run_layerwise_cached_kd_ablation. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.run_layerwise_cached_kd_ablation import *  # noqa: F403, F401
