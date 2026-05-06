"""Statistical tests for model comparison.

References:
  - Demšar (2006): Statistical Comparisons of Classifiers over Multiple Data Sets
  - Dunn (1961): Multiple Comparisons among Means
  - Nemenyi (1963): Distribution‑free Multiple Comparisons
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def anova_test(groups: Dict[str, np.ndarray]) -> Optional[Dict[str, float]]:
    """One‑way ANOVA across groups. Returns F‑stat and p‑value, or None if not applicable."""
    arrays = [v for v in groups.values() if len(v) > 1]
    if len(arrays) < 2:
        return None
    try:
        f_stat, p_val = stats.f_oneway(*arrays)
        return {"f_statistic": float(f_stat), "p_value": float(p_val)}
    except Exception as e:
        logger.warning("ANOVA failed: %s", e)
        return None


def pairwise_t_tests(
    groups: Dict[str, np.ndarray],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Paired t‑tests between all pairs of groups. Returns DataFrame of p‑values."""
    names = sorted(groups.keys())
    n = len(names)
    matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = groups[names[i]], groups[names[j]]
            if len(a) < 2 or len(b) < 2:
                continue
            t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
            # Bonferroni correction
            p_val *= n * (n - 1) / 2
            p_val = min(p_val, 1.0)
            matrix[i, j] = matrix[j, i] = p_val
    df = pd.DataFrame(matrix, index=names, columns=names)
    return df


def mann_whitney_test(
    groups: Dict[str, np.ndarray],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Mann‑Whitney U test (non‑parametric) between all pairs."""
    names = sorted(groups.keys())
    n = len(names)
    matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = groups[names[i]], groups[names[j]]
            try:
                u_stat, p_val = stats.mannwhitneyu(a, b, alternative='two-sided')
                p_val *= n * (n - 1) / 2
                p_val = min(p_val, 1.0)
                matrix[i, j] = matrix[j, i] = p_val
            except Exception:
                matrix[i, j] = matrix[j, i] = 1.0
    return pd.DataFrame(matrix, index=names, columns=names)


def nemenyi_test(
    groups: Dict[str, np.ndarray],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Nemenyi post‑hoc test for Friedman comparisons.
    Requires the same number of samples per group.
    """
    names = sorted(groups.keys())
    arrays = [groups[k] for k in names]
    # If unequal lengths, we cannot reliably perform Nemenyi (use Dunn instead)
    lengths = [len(a) for a in arrays]
    if len(set(lengths)) > 1:
        logger.info("Nemenyi test skipped – groups have unequal sample sizes. Use Dunn test.")
        return pd.DataFrame(np.ones((len(names), len(names))), index=names, columns=names)

    k = len(names)         # number of groups
    N = lengths[0]         # number of samples per group
    if k < 2 or N < 2:
        return pd.DataFrame(np.ones((k, k)), index=names, columns=names)

    try:
        # Friedman test first
        data = np.column_stack(arrays)
        friedman_stat, friedman_p = stats.friedmanchisquare(*arrays)

        # Critical value for Nemenyi at alpha
        from scipy.stats import studentized_range
        q_alpha = studentized_range.ppf(1 - alpha, k, np.inf)
        cd = q_alpha * np.sqrt(k * (k + 1) / (6 * N))

        # Rank each group
        ranks = np.zeros((N, k))
        for i in range(N):
            row = data[i]
            ranks[i] = stats.rankdata(row)   # 1 = best
        avg_ranks = np.mean(ranks, axis=0)

        # Pairwise differences
        matrix = np.ones((k, k))
        for i in range(k):
            for j in range(k):
                if i != j:
                    diff = abs(avg_ranks[i] - avg_ranks[j])
                    # Return the critical difference; p‑value not directly computable
                    matrix[i, j] = diff / cd   # >1 means significant difference

        df = pd.DataFrame(matrix, index=names, columns=names)
        return df
    except Exception as e:
        logger.warning("Nemenyi test failed: %s", e)
        return pd.DataFrame(np.ones((k, k)), index=names, columns=names)


def dunn_test(
    groups: Dict[str, np.ndarray],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Dunn's test – non‑parametric post‑hoc for Kruskal‑Wallis.
    Works with unequal sample sizes.
    """
    names = sorted(groups.keys())
    arrays = [groups[k] for k in names]
    k = len(names)
    N = sum(len(a) for a in arrays)

    if k < 2:
        return pd.DataFrame(np.ones((k, k)), index=names, columns=names)

    try:
        # Kruskal‑Wallis first
        h_stat, h_p = stats.kruskal(*arrays)

        # Rank all observations together
        all_data = np.concatenate(arrays)
        all_ranks = stats.rankdata(all_data)

        # Average rank per group
        start = 0
        avg_ranks = np.zeros(k)
        n_per_group = np.array([len(a) for a in arrays])
        for i, a in enumerate(arrays):
            avg_ranks[i] = np.mean(all_ranks[start:start + len(a)])
            start += len(a)

        # Pairwise z‑statistic
        matrix = np.ones((k, k))
        total = N * (N + 1) / 12
        for i in range(k):
            for j in range(i + 1, k):
                diff = abs(avg_ranks[i] - avg_ranks[j])
                se = np.sqrt(total * (1.0 / n_per_group[i] + 1.0 / n_per_group[j]))
                z = diff / se if se > 0 else 0
                # Two‑tailed p‑value from normal approximation
                p_val = 2 * (1 - stats.norm.cdf(abs(z)))
                p_val *= k * (k - 1) / 2   # Bonferroni
                p_val = min(p_val, 1.0)
                matrix[i, j] = matrix[j, i] = p_val
        return pd.DataFrame(matrix, index=names, columns=names)
    except Exception as e:
        logger.warning("Dunn test failed: %s", e)
        return pd.DataFrame(np.ones((k, k)), index=names, columns=names)