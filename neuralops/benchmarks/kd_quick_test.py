"""Deprecated shim for neuralops.benchmarks._legacy.kd_quick_test.

Use the unified CLI instead:
    python -m neuralops.benchmarks run ...

For programmatic use:
    from neuralops.benchmarks import benchmark, Runner
"""
import warnings
warnings.warn(
    "neuralops.benchmarks.kd_quick_test is deprecated and will be removed. "
    "Use  instead.",
    DeprecationWarning,
    stacklevel=2,
)
from neuralops.benchmarks._legacy.kd_quick_test import *  # noqa: F401,F403
try:
    from neuralops.benchmarks._legacy.kd_quick_test import main  # noqa: F401
except ImportError:
    pass
