"""Backward-compatibility shim.

This module has moved to neuralops.benchmarks.multi_mode_benchmark.
Update your imports:

    from neuralops.benchmarks.multi_mode_benchmark import MultiModeBenchmarkRunner
"""
import warnings

warnings.warn(
    "neuralops.core.multi_mode_benchmark has moved to "
    "neuralops.benchmarks.multi_mode_benchmark. Update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks.multi_mode_benchmark import *  # noqa: F401,F403
