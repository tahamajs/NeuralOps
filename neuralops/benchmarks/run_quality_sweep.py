"""Deprecated shim for neuralops.benchmarks.run_quality_sweep.

Use the unified CLI instead:
    python -m neuralops.benchmarks run ...
"""
import warnings

warnings.warn(
    "neuralops.benchmarks.run_quality_sweep is deprecated and will be removed. "
    "Use neuralops.benchmarks instead.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks.config import QualitySweepConfig

def run_quality_sweep(*args, **kwargs):
    """Legacy shim for running quality sweep benchmarks."""
    from neuralops.benchmarks.runner import BenchmarkRunner
    runner = BenchmarkRunner()
    return runner.run_all(*args, **kwargs)

try:
    from neuralops.benchmarks._legacy.run_quality_sweep import main  # noqa: F401
except ImportError:
    pass

__all__ = ["run_quality_sweep", "QualitySweepConfig"]
