"""Unit Tests for Advanced Statistical Analysis & Multiple-Comparison Corrections."""

import numpy as np
import pytest
from framework.run_quality_sweep import (compute_bh_fdr, compute_bootstrap_ci,
                                         compute_confidence_interval_95,
                                         compute_t_test_p_value)


def test_t_test_p_value_identical():
    # Identical performance should produce p=1.0
    accs1 = [85.0, 86.0, 87.0]
    accs2 = [85.0, 86.0, 87.0]
    p_val = compute_t_test_p_value(accs1, accs2)
    assert p_val == 1.0


def test_t_test_p_value_significant():
    # Large distinct difference should produce small p-value
    accs1 = [95.0, 96.0, 97.0]
    accs2 = [50.0, 51.0, 52.0]
    p_val = compute_t_test_p_value(accs1, accs2)
    assert p_val < 0.01


def test_bh_fdr_monotonicity():
    # B-H FDR adjusted p-values must be >= raw p-values and <= 1.0
    raw_p = [0.001, 0.01, 0.03, 0.05, 0.12, 0.50]
    bh_p = compute_bh_fdr(raw_p)
    assert len(bh_p) == len(raw_p)
    for r, b in zip(raw_p, bh_p):
        assert b >= r
        assert 0.0 <= b <= 1.0


def test_bh_fdr_empty():
    assert compute_bh_fdr([]) == []


def test_bootstrap_ci_bounds():
    accs = [82.5, 85.0, 87.5]
    ci_lower, ci_upper = compute_bootstrap_ci(accs, n_bootstraps=500, ci_level=0.95)
    mean_val = np.mean(accs)
    assert ci_lower <= mean_val <= ci_upper
    assert ci_lower >= 0.0
    assert ci_upper <= 100.0


def test_confidence_interval_95_small_sample():
    accs = [80.0, 85.0, 90.0]
    ci_lower, ci_upper = compute_confidence_interval_95(accs)
    assert ci_lower < 85.0 < ci_upper
