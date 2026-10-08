"""Statistical comparison utilities for benchmark analysis."""
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

__all__ = [
    "holm_correction",
    "hochberg_correction",
    "finner_correction",
    "hommel_correction",
    "holland_copenhaver_correction",
    "wilcoxon_signed_rank_test",
    "sign_test",
    "mcnemar_test",
    "bayesian_correlated_ttest",
    "bayesian_signed_rank_test",
    "run_full_statistical_analysis",
]
