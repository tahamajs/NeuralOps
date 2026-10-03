Module neuralops.analysis.comparison.statistical_tests
======================================================
Complete statistical tests for model comparison – publication‑grade.

References:
  - Demšar (2006): Statistical Comparisons of Classifiers over Multiple Data Sets (JMLR)
  - Benavoli et al. (2017): Time for a Change – Bayesian Comparison of Classifiers (JMLR)
  - García & Herrera (2008): An Extension on Statistical Comparisons of Classifiers
  - García et al. (2010): Advanced nonparametric tests for multiple comparisons
  - Derrac et al. (2011): A practical tutorial on the use of nonparametric tests
  - Iman & Davenport (1980): Approximations of the critical region for Friedman
  - Quade (1979): Using weighted rankings in analysis of complete blocks
  - Cliff (1993): Dominance statistics – ordinal analyses to answer ordinal questions
  - Vargha & Delaney (2000): Critique and improvement of CL common language effect size
  - Cohen (1988): Statistical Power Analysis for the Behavioral Sciences
  - Hedges & Olkin (1985): Statistical Methods for Meta-Analysis
  - Nemenyi (1963): Distribution-free Multiple Comparisons
  - Dunn (1961): Multiple Comparisons among Means
  - Holm (1979): A simple sequentially rejective multiple test procedure
  - Hochberg (1988): A sharper Bonferroni procedure
  - Hommel (1988): A stagewise rejective multiple test procedure
  - Holland & Copenhaver (1987): An improved sequentially rejective Bonferroni test
  - Finner (1993): On a monotonicity problem in step-down multiple test procedures
  - Bergmann & Hommel (1988): Improvements of general multiple test procedures

Functions
---------

`bayesian_correlated_ttest(x: numpy.ndarray, y: numpy.ndarray, rope: float = 0.01, runs: int = 1, nsamples: int = 50000, seed: int = 42) ‑> dict[str, float]`
:   Bayesian correlated t‑test for k‑fold cross‑validation results.
    
    Based on: Benavoli et al. (2017) "Time for a Change"
    ROPE = Region of Practical Equivalence
    
    Returns probabilities: p_left (x < y), p_rope (x ≈ y), p_right (x > y).
    
    Note: This is a simplified implementation using a Monte Carlo approximation.
    For exact results, use the `baycomp` Python package.

`bayesian_signed_rank_test(x: numpy.ndarray, y: numpy.ndarray, rope: float = 0.01, nsamples: int = 50000, seed: int = 42) ‑> dict[str, float]`
:   Bayesian signed‑rank test – Bayesian equivalent of Wilcoxon.
    
    Based on: Benavoli et al. (2014, 2017)
    Simplified implementation using Dirichlet process prior approximation.

`bergmann_hommel_posthoc(avg_ranks: numpy.ndarray, n_datasets: int, k: int = None, alpha: float = 0.05) ‑> pandas.DataFrame`
:   Bergmann-Hommel post-hoc – the most powerful post-hoc for Friedman test.
    Uses the logical structure of hypotheses for dynamic correction.
    
    References: Bergmann & Hommel (1988), García & Herrera (2008)

`bonferroni_dunn_posthoc(p_values: numpy.ndarray | None = None, k: int = 0, n: int = 0, alpha: float = 0.05) ‑> dict[str, typing.Any]`
:   Bonferroni-Dunn post-hoc: control method vs. all others.
    
    Adjusted alpha = alpha / (k-1)
    
    Reference: Dunn (1961)

`cliffs_delta(a: numpy.ndarray, b: numpy.ndarray) ‑> dict[str, float]`
:   Cliff's delta – non‑parametric effect size for ordinal data.
    Measures the probability that a randomly selected value from group A
    exceeds a randomly selected value from group B.
    
    delta = (#(a > b) - #(a < b)) / (n_a * n_b)
    
    Reference: Cliff (1993)

`cohens_d(a: numpy.ndarray, b: numpy.ndarray, paired: bool = True) ‑> dict[str, float]`
:   Cohen's d – standardised mean difference.
    
    Reference: Cohen (1988)

`compute_aligned_ranks(scores_matrix: numpy.ndarray) ‑> numpy.ndarray`
:   Compute aligned ranks (Hodges-Lehmann / Friedman Aligned Ranks).
    
    The aligned rank for each observation is its rank among ALL observations
    after subtracting the row mean.  This allows comparison across rows.
    
    Reference: Hodges & Lehmann (1962)

`compute_ranks(scores_matrix: numpy.ndarray) ‑> numpy.ndarray`
:   Compute average ranks per column (algorithm) across rows (datasets).
    Within each row, the best algorithm gets rank 1 (highest score = best).
    
    Args:
        scores_matrix: shape (n_datasets, n_algorithms) – higher is better
    
    Returns:
        ranks: shape (n_algorithms,) – average rank

`critical_difference(k: int, N: int, alpha: float = 0.05) ‑> float`
:   Nemenyi Critical Difference (CD).
    
    CD = q_alpha * sqrt(k * (k + 1) / (6 * N))
    
    Two algorithms are significantly different if their average rank
    difference exceeds CD.
    
    Reference: Nemenyi (1963), Demšar (2006)

`finner_correction(p_values: list[float] | numpy.ndarray) ‑> numpy.ndarray`
:   Finner step‑down correction – most powerful but less common.
    
    Reference: Finner (1993)

`friedman_aligned_ranks_test(scores_matrix: numpy.ndarray) ‑> dict[str, float]`
:   Friedman Aligned Ranks test – more powerful than the standard Friedman test.
    
    Reference: García et al. (2010), Hodges & Lehmann (1962)

`friedman_test(scores_matrix: numpy.ndarray) ‑> dict[str, float]`
:   Friedman test – the classic omnibus test for comparing k algorithms
    over N datasets.  H0: all algorithms perform equivalently.

`glass_rank_biserial(a: numpy.ndarray, b: numpy.ndarray) ‑> dict[str, float]`
:   Glass rank-biserial correlation – effect size for Mann‑Whitney / Wilcoxon.
    r_rb = 2 * (A12 - 0.5)
    
    Reference: Glass (1966)

`hedges_g(a: numpy.ndarray, b: numpy.ndarray, paired: bool = True) ‑> dict[str, float]`
:   Hedges' g – bias‑corrected Cohen's d for small samples.
    
    Reference: Hedges & Olkin (1985)

`hochberg_correction(p_values: list[float] | numpy.ndarray) ‑> numpy.ndarray`
:   Hochberg step‑up correction – generally more powerful than Holm.
    
    Reference: Hochberg (1988)

`holland_copenhaver_correction(p_values: list[float] | numpy.ndarray) ‑> numpy.ndarray`
:   Holland & Copenhaver correction – improved sequentially rejective Bonferroni.
    
    Reference: Holland & Copenhaver (1987)

`holm_correction(p_values: list[float] | numpy.ndarray) ‑> numpy.ndarray`
:   Holm step-down correction – controls family-wise error rate.
    More powerful than Bonferroni.
    
    Reference: Holm (1979)

`hommel_correction(p_values: list[float] | numpy.ndarray) ‑> numpy.ndarray`
:   Hommel correction – stagewise rejective procedure.
    
    Reference: Hommel (1988)

`mcnemar_test(y_true: numpy.ndarray, pred_a: numpy.ndarray, pred_b: numpy.ndarray) ‑> dict[str, typing.Any]`
:   McNemar's test for comparing two classifiers on a single test set.
    Uses a 2x2 contingency table of correct/incorrect predictions.
    
    Reference: McNemar (1947)

`nemenyi_posthoc(avg_ranks: numpy.ndarray, n_datasets: int, alpha: float = 0.05) ‑> pandas.DataFrame`
:   Nemenyi post-hoc test: all vs. all pairwise comparisons.
    
    Returns a DataFrame of p-values (approximated from CD).

`quade_test(scores_matrix: numpy.ndarray) ‑> dict[str, float]`
:   Quade test – weighted ranking test that gives more weight to datasets
    where larger differences between algorithms exist.
    
    References: Quade (1979), Conover (1999)

`run_full_statistical_analysis(scores_matrix: numpy.ndarray, algorithm_names: list[str] | None = None, alpha: float = 0.05, generate_posthoc: bool = True) ‑> dict[str, typing.Any]`
:   Run the complete statistical pipeline recommended by Demšar (2006)
    and extended by García & Herrera (2008, 2010):
    
    1. Friedman test (with Iman-Davenport correction)
    2. Friedman Aligned Ranks test
    3. Quade test
    4. If significant: Nemenyi post‑hoc + Bonferroni-Dunn + Holm + Hochberg
    5. Average ranks + ranking
    
    Args:
        scores_matrix: shape (n_datasets, n_algorithms), higher = better
        algorithm_names: list of algorithm name strings
        alpha: significance level
        generate_posthoc: whether to compute post‑hoc tests
    
    Returns:
        Dictionary with all results

`sign_test(scores_a: numpy.ndarray, scores_b: numpy.ndarray) ‑> dict[str, typing.Any]`
:   Simple sign test – counts wins/losses with binomial distribution.
    Less powerful than Wilcoxon but assumption‑free.

`vargha_delaney_a12(a: numpy.ndarray, b: numpy.ndarray) ‑> dict[str, float]`
:   Vargha-Delaney's A12 – stochastic superiority measure.
    A12 = P(X > Y) + 0.5 * P(X = Y)
    
    Interpretation: 0.56=small, 0.64=medium, 0.71=large
    
    Reference: Vargha & Delaney (2000)

`wilcoxon_signed_rank_test(scores_a: numpy.ndarray, scores_b: numpy.ndarray) ‑> dict[str, typing.Any]`
:   Wilcoxon signed-rank test – the recommended test for comparing
    two classifiers over multiple datasets (Demšar 2006).