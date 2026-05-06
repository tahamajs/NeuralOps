import numpy as np
from scipy import stats
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

@dataclass
class ABTestResult:
    metric: str; mean_a: float; mean_b: float
    difference: float; p_value: float; significant: bool
    better_model: str; confidence_interval: Tuple[float, float]

class ABTester:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def compare(self, results_a: Dict[str, List[float]],
                results_b: Dict[str, List[float]]) -> List[ABTestResult]:
        outcomes = []
        for metric in results_a:
            if metric not in results_b: continue
            a = np.array(results_a[metric]); b = np.array(results_b[metric])
            t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
            mean_a, mean_b = np.mean(a), np.mean(b)
            diff = mean_b - mean_a
            se = np.sqrt(np.var(a)/len(a) + np.var(b)/len(b))
            ci = (diff - 1.96*se, diff + 1.96*se)
            outcomes.append(ABTestResult(
                metric=metric, mean_a=mean_a, mean_b=mean_b,
                difference=diff, p_value=p_val,
                significant=p_val < self.alpha,
                better_model="B" if diff > 0 else "A",
                confidence_interval=ci
            ))
        return outcomes