"""Statistical significance testing, bootstrap intervals, and FDR correction for benchmarks."""
from __future__ import annotations

import numpy as np
from scipy import stats


def paired_ttest(accs1: list[float], accs2: list[float]) -> float:
    """
    Compute two-tailed Welch's t-test p-value between two distribution samples.

    Args:
        accs1: First sample observations.
        accs2: Second sample observations.

    Returns:
        P-value between 0.0 and 1.0.
    """
    arr1, arr2 = np.array(accs1, dtype=np.float64), np.array(accs2, dtype=np.float64)
    if np.array_equal(arr1, arr2) or (np.var(arr1) == 0 and np.var(arr2) == 0):
        return 1.0
    try:
        res = stats.ttest_ind(arr1, arr2, equal_var=False)
        p = float(res.pvalue)
        return 1.0 if np.isnan(p) else p
    except Exception:
        return 1.0


def bootstrap_ci(
    accs: list[float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Compute non-parametric bootstrap confidence interval for the sample mean.

    Args:
        accs: Sample observation scores.
        n_boot: Number of bootstrap resamples.
        ci: Desired confidence level (default: 0.95).
        seed: Random seed for bootstrap generator.

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    arr = np.array(accs, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0
    if len(arr) == 1 or np.var(arr) == 0:
        val = float(arr[0])
        return val, val

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot_means = np.mean(arr[indices], axis=1)

    alpha = 1.0 - ci
    lower = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    upper = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))
    return lower, upper


def fdr_correction(raw_p: list[float], alpha: float = 0.05) -> list[float]:
    """
    Benjamini-Hochberg False Discovery Rate (FDR) adjusted p-values.

    Args:
        raw_p: List of unadjusted p-values.
        alpha: Target false discovery rate threshold.

    Returns:
        List of monotonic FDR-adjusted p-values matching input order.
    """
    n = len(raw_p)
    if n == 0:
        return []
    sorted_indices = np.argsort(raw_p)
    sorted_p = np.array(raw_p)[sorted_indices]
    adj_p = np.zeros(n)

    cummin = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        p_adj = (sorted_p[i] * n) / rank
        cummin = min(cummin, p_adj)
        adj_p[i] = min(cummin, 1.0)

    result = np.zeros(n)
    result[sorted_indices] = adj_p
    return result.tolist()


__all__ = [
    "paired_ttest",
    "bootstrap_ci",
    "fdr_correction",
]
