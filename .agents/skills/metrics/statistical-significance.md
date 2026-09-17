# Statistical Significance

## Purpose

Every accuracy claim must be backed by a p-value and effect size.

## Tests to run

| Test | When to use |
|---|---|
| Paired t-test | Same eval set, both models evaluated |
| McNemar's test | Binary classification, per-example predictions |
| Wilcoxon signed-rank | Non-normal distributions |
| Bootstrap CI | Always; 1000 resamples |

## Implementation

```python
import numpy as np
from scipy import stats

def paired_ttest(acc_a, acc_b):
    """acc_a, acc_b: list of per-seed accuracies."""
    t, p = stats.ttest_rel(acc_a, acc_b)
    return {"t": t, "p": p}

def cohens_d(a, b):
    diff = np.array(a) - np.array(b)
    return diff.mean() / (diff.std(ddof=1) + 1e-12)

def bootstrap_ci(values, n=1000, alpha=0.05):
    samples = np.random.choice(values, (n, len(values)), replace=True)
    means = samples.mean(axis=1)
    return np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
```

## Output

- `papers/modules/<module>/metrics/significance_tests.csv`
- Columns: module, backbone, rank, metric, baseline_mean, module_mean, delta, t_stat, p_value, cohens_d, significant_at_0.05
- `papers/modules/<module>/metrics/bootstrap.json`
- `papers/modules/<module>/metrics/effect_size.csv`

## Rules

- Difference below 2×SE is noise, not improvement
- Report effect size, not just p-value
- Correct for multiple comparisons (Bonferroni or FDR)
