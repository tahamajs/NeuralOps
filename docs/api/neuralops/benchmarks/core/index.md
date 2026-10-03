Module neuralops.benchmarks.core
================================
Shared core utilities for the NeuralOps benchmarks suite.

Modules
-------
arguments
    Unified CLI argument parser (``parse_args``) used by all benchmark drivers.
stats
    Statistical helpers: t-tests, bootstrap confidence intervals, BH-FDR correction.

Sub-modules
-----------
* neuralops.benchmarks.core.arguments
* neuralops.benchmarks.core.stats

Functions
---------

`compute_bh_fdr(raw_p: list[float]) ‑> list[float]`
:   

`compute_bootstrap_ci(accs: list[float], n_bootstraps: int = 500, ci_level: float = 0.95) ‑> tuple[float, float]`
:   

`compute_confidence_interval_95(accs: list[float]) ‑> tuple[float, float]`
:   

`compute_t_test_p_value(accs1: list[float], accs2: list[float]) ‑> float`
:   

`parse_args(argv=None)`
: