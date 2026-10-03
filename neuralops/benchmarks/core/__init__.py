"""
Shared core utilities for the NeuralOps benchmarks suite.

Modules
-------
arguments
    Unified CLI argument parser (``parse_args``) used by all benchmark drivers.
stats
    Statistical helpers: confidence intervals, Wilcoxon tests, effect sizes.
"""

from .arguments import parse_args
from .stats import compute_confidence_interval, compute_effect_size, wilcoxon_test

__all__ = [
    "parse_args",
    "compute_confidence_interval",
    "compute_effect_size",
    "wilcoxon_test",
]
