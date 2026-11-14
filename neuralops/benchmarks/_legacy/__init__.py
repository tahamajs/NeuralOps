"""Quarantined legacy benchmark scripts and entrypoints.

All scripts in this package are deprecated and scheduled for removal.
Please migrate to canonical NeuralOps CLI v2 commands ('neuralops benchmark').
"""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks._legacy is deprecated and will be removed in a future release. "
    "Use canonical NeuralOps CLI v2 ('neuralops benchmark') instead.",
    DeprecationWarning,
    stacklevel=2,
)
