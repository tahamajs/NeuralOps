#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Module providing ab_testing implementations for NeuralOps.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy import stats


@dataclass
class ABTestResult:
    """Abtestresult definition."""
    metric: str
    mean_a: float
    mean_b: float
    difference: float
    p_value: float
    significant: bool
    better_model: str
    confidence_interval: tuple[float, float]


class ABTester:
    """Abtester definition."""
    def __init__(self, alpha: float = 0.05) -> None:
        """Execute __init__ operation."""
        self.alpha = alpha

    def compare(
        self, results_a: dict[str, list[float]], results_b: dict[str, list[float]]
    ) -> list[ABTestResult]:
        """Compare."""
        outcomes = []
        for metric in results_a:
            if metric not in results_b:
                continue
            a = np.array(results_a[metric])
            b = np.array(results_b[metric])
            t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
            mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
            diff = mean_b - mean_a
            n1, n2 = len(a), len(b)
            v1 = float(np.var(a, ddof=1)) if n1 > 1 else float(np.var(a))
            v2 = float(np.var(b, ddof=1)) if n2 > 1 else float(np.var(b))
            se = math.sqrt(v1 / n1 + v2 / n2) if (n1 > 0 and n2 > 0) else 0.0
            num = (v1 / n1 + v2 / n2) ** 2
            den = ((v1 / n1) ** 2) / (n1 - 1) + ((v2 / n2) ** 2) / (n2 - 1) if (n1 > 1 and n2 > 1) else 1.0
            df = num / den if den > 0 else float(n1 + n2 - 2)
            t_crit = float(stats.t.ppf(1.0 - self.alpha / 2.0, df=df)) if df > 0 else 1.96
            ci = (float(diff - t_crit * se), float(diff + t_crit * se))
            outcomes.append(
                ABTestResult(
                    metric=metric,
                    mean_a=mean_a,
                    mean_b=mean_b,
                    difference=diff,
                    p_value=p_val,
                    significant=p_val < self.alpha,
                    better_model="B" if diff > 0 else "A",
                    confidence_interval=ci,
                )
            )
        return outcomes
