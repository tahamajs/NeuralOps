"""
Shared core utilities for the NeuralOps benchmarks suite.

Modules
-------
arguments
    Unified CLI argument parser (``parse_args``) used by all benchmark drivers.
stats
    Statistical helpers: t-tests, bootstrap confidence intervals, BH-FDR correction.
"""

from .arguments import parse_args
from .stats import (
    compute_bh_fdr,
    compute_bootstrap_ci,
    compute_confidence_interval_95,
    compute_t_test_p_value,
)

__all__ = [
    "parse_args",
    "compute_bootstrap_ci",
    "compute_bh_fdr",
    "compute_confidence_interval_95",
    "compute_t_test_p_value",
]
