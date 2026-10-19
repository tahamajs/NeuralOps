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
            mean_a, mean_b = np.mean(a), np.mean(b)
            diff = mean_b - mean_a
            se = np.sqrt(np.var(a) / len(a) + np.var(b) / len(b))
            ci = (diff - 1.96 * se, diff + 1.96 * se)
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
