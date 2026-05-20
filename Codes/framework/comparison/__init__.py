"""
Comparison engine and statistical tests for multi‑run analysis.
"""

from .engine import ComparisonEngine, ComparisonResult
from .leaderboard import generate_leaderboard
from .statistical_tests import (
    anova_test, pairwise_t_tests, mann_whitney_test,
    nemenyi_test, dunn_test,
    # Omnibus
    friedman_test, friedman_aligned_ranks_test, quade_test,
    # Post-hoc
    critical_difference, nemenyi_posthoc, bonferroni_dunn_posthoc,
    holm_correction, hochberg_correction, finner_correction,
    hommel_correction, holland_copenhaver_correction,
    bergmann_hommel_posthoc,
    # Pairwise
    wilcoxon_signed_rank_test, sign_test, mcnemar_test,
    # Effect sizes
    cohens_d, hedges_g, cliffs_delta, vargha_delaney_a12, glass_rank_biserial,
    # Bayesian
    bayesian_correlated_ttest, bayesian_signed_rank_test,
    # Convenience
    run_full_statistical_analysis, compute_ranks,
)
from .plotting import plot_critical_difference_diagram, plot_pairwise_comparison_heatmap

__all__ = [
    "ComparisonEngine",
    "ComparisonResult",
    "generate_leaderboard",
    "anova_test",
    "pairwise_t_tests",
    "mann_whitney_test",
    "nemenyi_test",
    "dunn_test",
    "friedman_test",
    "friedman_aligned_ranks_test",
    "quade_test",
    "critical_difference",
    "nemenyi_posthoc",
    "bonferroni_dunn_posthoc",
    "holm_correction",
    "hochberg_correction",
    "finner_correction",
    "hommel_correction",
    "holland_copenhaver_correction",
    "bergmann_hommel_posthoc",
    "wilcoxon_signed_rank_test",
    "sign_test",
    "mcnemar_test",
    "cohens_d",
    "hedges_g",
    "cliffs_delta",
    "vargha_delaney_a12",
    "glass_rank_biserial",
    "bayesian_correlated_ttest",
    "bayesian_signed_rank_test",
    "run_full_statistical_analysis",
    "compute_ranks",
    "plot_critical_difference_diagram",
    "plot_pairwise_comparison_heatmap",
]