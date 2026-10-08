"""Deprecated shim for neuralops.benchmarks._legacy.run_multiseed.

Use the unified CLI instead:
    python -m neuralops.benchmarks run ...

For programmatic use:
    from neuralops.benchmarks import benchmark, Runner
"""
import warnings
warnings.warn(
    "neuralops.benchmarks.run_multiseed is deprecated and will be removed. "
    "Use  instead.",
    DeprecationWarning,
    stacklevel=2,
)
from neuralops.benchmarks._legacy.run_multiseed import *  # noqa: F401,F403
try:
    from neuralops.benchmarks._legacy.run_multiseed import main  # noqa: F401
except ImportError:
    pass
