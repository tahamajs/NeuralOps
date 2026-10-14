"""Deprecated shim for neuralops.benchmarks.run_gpt2_sweep.

Use the unified CLI instead:
    python -m neuralops.benchmarks run ...
"""
import warnings

warnings.warn(
    "neuralops.benchmarks.run_gpt2_sweep is deprecated and will be removed. "
    "Use neuralops.benchmarks instead.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks.config import GPT2SweepConfig

def run_gpt2_sweep(*args, **kwargs):
    """Legacy shim for running GPT-2 benchmarks."""
    from neuralops.benchmarks.runner import BenchmarkRunner
    runner = BenchmarkRunner()
    return runner.run_all(*args, **kwargs)

__all__ = ["run_gpt2_sweep", "GPT2SweepConfig"]
