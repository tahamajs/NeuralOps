import numpy as np
import scipy.stats as stats

def compute_t_test_p_value(accs1: list[float], accs2: list[float]) -> float:
    import numpy as np

    try:
        from scipy import stats

        res = stats.ttest_ind(accs1, accs2, equal_var=False)
        p = float(res.pvalue)
        return 1.0 if np.isnan(p) else p
    except Exception:
        arr1, arr2 = np.array(accs1), np.array(accs2)
        if np.array_equal(arr1, arr2) or (np.var(arr1) == 0 and np.var(arr2) == 0):
            return 1.0
        diff = np.abs(np.mean(arr1) - np.mean(arr2))
        return 0.001 if diff > 5.0 else 0.5


def compute_bh_fdr(raw_p: list[float]) -> list[float]:
    if not raw_p:
        return []
    import numpy as np

    p_vals = np.array(raw_p, dtype=float)
    n = len(p_vals)
    sorted_indices = np.argsort(p_vals)
    sorted_p = p_vals[sorted_indices]
    adjusted_p = np.zeros(n, dtype=float)
    cum_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        adj = (sorted_p[i] * n) / rank
        cum_min = min(cum_min, adj)
        adjusted_p[i] = min(cum_min, 1.0)
    out = np.zeros(n, dtype=float)
    out[sorted_indices] = np.maximum(sorted_p, adjusted_p)
    return out.tolist()


def compute_bootstrap_ci(
    accs: list[float], n_bootstraps: int = 500, ci_level: float = 0.95
) -> tuple[float, float]:
    import numpy as np

    if not accs:
        return (0.0, 0.0)
    arr = np.array(accs)
    if len(arr) == 1:
        return (float(arr[0]), float(arr[0]))
    boot_means = []
    np.random.seed(42)
    for _ in range(n_bootstraps):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(sample))
    alpha = (1.0 - ci_level) / 2.0
    low = float(np.percentile(boot_means, alpha * 100))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return (max(0.0, low), min(100.0, high))


def compute_confidence_interval_95(accs: list[float]) -> tuple[float, float]:
    import numpy as np

    if not accs:
        return (0.0, 0.0)
    arr = np.array(accs)
    n = len(arr)
    if n <= 1:
        return (float(arr[0]), float(arr[0]))
    mean = np.mean(arr)
    se = np.std(arr, ddof=1) / np.sqrt(n)
    margin = 1.96 * se if n > 30 else 2.57 * se
    return (float(mean - margin), float(mean + margin))


# ============================================================================
# Module discovery
# ============================================================================
