"""Deprecation forwarding shim for run_parallel_vram_saturation_bench."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.run_parallel_vram_saturation_bench is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.run_parallel_vram_saturation_bench. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.run_parallel_vram_saturation_bench import *  # noqa: F403, F401
