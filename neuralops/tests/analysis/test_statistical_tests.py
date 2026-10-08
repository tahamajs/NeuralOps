"""Tests for neuralops.analysis.comparison.statistical_tests."""

import numpy as np
import pytest

from neuralops.analysis.comparison import (
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


class TestMultipleComparisonCorrections:
    """Test family-wise error rate and step-up / step-down corrections."""

    @pytest.mark.parametrize(
        "correction_fn",
        [
            holm_correction,
            hochberg_correction,
            finner_correction,
            hommel_correction,
            holland_copenhaver_correction,
        ],
    )
    def test_correction_increases_pvalues(self, correction_fn):
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        corrected = correction_fn(p_values)
        assert len(corrected) == len(p_values)
        for orig, corr in zip(p_values, corrected):
            assert corr >= orig - 1e-10

    @pytest.mark.parametrize(
        "correction_fn",
        [
            holm_correction,
            hochberg_correction,
            finner_correction,
        ],
    )
    def test_correction_bounds(self, correction_fn):
        p_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        corrected = correction_fn(p_values)
        assert all(0 <= p <= 1 for p in corrected)

    def test_holm_monotonicity(self):
        p_values = [0.001, 0.01, 0.05]
        corrected = holm_correction(p_values)
        assert sorted(corrected) == list(corrected)


class TestHypothesisTests:
    """Test Wilcoxon, Sign test, and McNemar test."""

    def test_wilcoxon_identical_samples(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = wilcoxon_signed_rank_test(x, y)
        assert "p_value" in result
        assert result["p_value"] == 1.0

    def test_wilcoxon_different_samples(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = wilcoxon_signed_rank_test(x, y)
        assert result["p_value"] < 0.1

    def test_sign_test(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        y = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        result = sign_test(x, y)
        assert "p_value" in result
        assert result["wins_b"] == 8

    def test_mcnemar(self):
        y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
        pred_a = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 1])
        pred_b = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1])
        result = mcnemar_test(y_true, pred_a, pred_b)
        assert "chi2" in result
        assert "p_value" in result


class TestBayesianTests:
    """Test Bayesian correlated t-test and signed-rank test."""

    def test_bayesian_ttest_basic(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        result = bayesian_correlated_ttest(x, y, nsamples=1000)
        assert "p_left" in result
        assert "p_rope" in result
        assert "p_right" in result

    def test_bayesian_signed_rank(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1.5, 2.5, 3.5, 4.5])
        result = bayesian_signed_rank_test(x, y, nsamples=1000)
        assert "p_left" in result
        assert "p_rope" in result
        assert "p_right" in result


class TestFullAnalysis:
    """Test full multi-algorithm comparative statistical analysis pipeline."""

    def test_run_full_statistical_analysis(self):
        # shape (n_datasets=5, n_algorithms=3)
        scores_matrix = np.array([
            [0.85, 0.80, 0.75],
            [0.86, 0.82, 0.77],
            [0.84, 0.81, 0.76],
            [0.87, 0.83, 0.78],
            [0.85, 0.80, 0.75],
        ])
        output = run_full_statistical_analysis(
            scores_matrix, algorithm_names=["baseline", "mod_a", "mod_b"]
        )
        assert "friedman" in output
        assert "quade" in output
        assert output["best_algorithm"] == "baseline"
