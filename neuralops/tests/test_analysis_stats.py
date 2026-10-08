"""Unit tests for statistical correction and hypothesis testing utilities."""
from __future__ import annotations

import numpy as np
import pytest

from neuralops.analysis.comparison.statistical_tests import (
    bayesian_correlated_ttest,
    bayesian_signed_rank_test,
    finner_correction,
    hochberg_correction,
    holland_copenhaver_correction,
    holm_correction,
    hommel_correction,
    mcnemar_test,
    run_full_statistical_analysis,
    sign_test,
    wilcoxon_signed_rank_test,
)


class TestMultipleTestingCorrections:
    """Test family-wise error rate and false discovery rate corrections."""

    def test_holm_correction(self):
        p_vals = np.array([0.01, 0.04, 0.03, 0.005])
        adj_p = holm_correction(p_vals)
        assert len(adj_p) == len(p_vals)
        assert np.all(adj_p >= p_vals)
        assert np.all(adj_p <= 1.0)

    def test_hochberg_correction(self):
        p_vals = np.array([0.001, 0.02, 0.08])
        adj_p = hochberg_correction(p_vals)
        assert len(adj_p) == len(p_vals)
        assert adj_p[0] < adj_p[2]

    def test_finner_correction(self):
        p_vals = np.array([0.002, 0.01, 0.05, 0.2])
        adj_p = finner_correction(p_vals)
        assert len(adj_p) == len(p_vals)
        assert np.all(adj_p >= 0.0)
        assert np.all(adj_p <= 1.0)

    def test_hommel_correction(self):
        p_vals = np.array([0.001, 0.01, 0.04, 0.1])
        adj_p = hommel_correction(p_vals)
        assert len(adj_p) == len(p_vals)
        assert np.all(adj_p <= 1.0)

    def test_holland_copenhaver_correction(self):
        p_vals = np.array([0.005, 0.02, 0.05])
        adj_p = holland_copenhaver_correction(p_vals)
        assert len(adj_p) == len(p_vals)
        assert np.all(adj_p >= 0.0)


class TestHypothesisTests:
    """Test non-parametric paired tests and contingency table tests."""

    def test_wilcoxon_signed_rank_test(self):
        x = np.array([1.2, 2.5, 3.1, 4.0, 5.2, 6.1])
        y = np.array([1.0, 2.2, 2.9, 3.8, 5.0, 5.9])
        res = wilcoxon_signed_rank_test(x, y)
        assert isinstance(res, dict)
        assert "statistic" in res
        assert "p_value" in res
        assert 0.0 <= res["p_value"] <= 1.0

    def test_sign_test(self):
        x = np.array([10.0, 12.0, 15.0, 14.0, 16.0])
        y = np.array([9.0, 11.0, 13.0, 13.5, 15.5])
        res = sign_test(x, y)
        assert isinstance(res, dict)
        assert "wins_a" in res
        assert "wins_b" in res
        assert 0.0 <= res["p_value"] <= 1.0

    def test_mcnemar_test(self):
        y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
        pred_a = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 1])
        pred_b = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1])
        res = mcnemar_test(y_true, pred_a, pred_b)
        assert isinstance(res, dict)
        assert "chi2" in res
        assert "p_value" in res
        assert 0.0 <= res["p_value"] <= 1.0

    def test_bayesian_correlated_ttest(self):
        x = np.array([0.85, 0.86, 0.84, 0.87, 0.85])
        y = np.array([0.82, 0.83, 0.81, 0.84, 0.82])
        res = bayesian_correlated_ttest(x, y)
        assert isinstance(res, dict)
        assert len(res) > 0

    def test_bayesian_signed_rank_test(self):
        x = np.array([0.85, 0.86, 0.84, 0.87, 0.85])
        y = np.array([0.82, 0.83, 0.81, 0.84, 0.82])
        res = bayesian_signed_rank_test(x, y)
        assert isinstance(res, dict)
        assert len(res) > 0

    def test_run_full_statistical_analysis(self):
        scores = np.array([
            [0.85, 0.82, 0.80],
            [0.86, 0.83, 0.81],
            [0.84, 0.81, 0.79],
            [0.87, 0.84, 0.82],
        ])
        names = ["model_a", "model_b", "model_c"]
        res = run_full_statistical_analysis(scores, algorithm_names=names)
        assert res is not None
        assert isinstance(res, dict)

    def test_edge_cases_empty_or_single(self):
        res = hommel_correction(np.array([0.05]))
        assert len(res) == 1
        assert res[0] == 0.05
