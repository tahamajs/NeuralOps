"""Backward-compatibility shim.

Moved to neuralops.core._archive.legacy_ensemble.
"""
import warnings

warnings.warn(
    "neuralops.core.legacy_ensemble is archived. "
    "Use neuralops.core._archive.legacy_ensemble if needed.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.core._archive.legacy_ensemble import *  # noqa: F401,F403
from neuralops.core._archive.legacy_ensemble import (  # noqa: F401
    SmartWeightedEnsemble,
    TaskAwareAttention,
)
