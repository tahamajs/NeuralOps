"""Deprecated shim for neuralops.experiment_utils.main.

This module is a duplicate of main.py. Use main.py instead:
    from neuralops.experiment_utils.main import ...
"""
import warnings
warnings.warn(
    "neuralops.experiment_utils.runner is a duplicate of main. "
    "Use neuralops.experiment_utils.main instead.",
    DeprecationWarning,
    stacklevel=2,
)
from neuralops.experiment_utils.main import *  # noqa: F401,F403
