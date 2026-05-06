# framework/evaluation/__init__.py
from .fairness import FairnessEvaluator, FairnessReport
from .robustness import RobustnessTester, RobustnessReport

# framework/evaluation/fairness.py
import numpy as np
from sklearn.metrics import accuracy_score
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

@dataclass
class FairnessReport:
    metric: str; overall: float
    group_values: Dict[str, float]; disparities: Dict[str, float]
    worst_group: str; best_group: str
    max_disparity: float; is_fair: bool; threshold: float

class FairnessEvaluator:
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    def evaluate(self, predictions: np.ndarray, labels: np.ndarray,
                 groups: Dict[str, np.ndarray],
                 metrics: List[str] = None) -> List[FairnessReport]:
        if metrics is None: metrics = ["accuracy", "demographic_parity", "equal_opportunity"]
        reports = []
        for metric in metrics:
            for group_name, group_values in groups.items():
                unique_groups = np.unique(group_values)
                group_scores = {}
                for g in unique_groups:
                    mask = group_values == g
                    if metric == "accuracy":
                        group_scores[str(g)] = accuracy_score(labels[mask], predictions[mask])
                    elif metric == "demographic_parity":
                        group_scores[str(g)] = np.mean(predictions[mask] == 1) if predictions.ndim == 1 \
                            else np.mean(np.argmax(predictions[mask], axis=1) == 1)
                    elif metric == "equal_opportunity":
                        pos_mask = labels[mask] == 1
                        if pos_mask.sum() > 0:
                            group_scores[str(g)] = accuracy_score(
                                labels[mask][pos_mask], predictions[mask][pos_mask])
                        else: group_scores[str(g)] = 0.0

                overall = np.mean(list(group_scores.values()))
                disparities = {k: abs(v - overall) for k, v in group_scores.items()}
                max_disp = max(disparities.values()) if disparities else 0
                reports.append(FairnessReport(
                    metric=metric, overall=overall, group_values=group_scores,
                    disparities=disparities,
                    worst_group=min(group_scores, key=group_scores.get),
                    best_group=max(group_scores, key=group_scores.get),
                    max_disparity=max_disp,
                    is_fair=max_disp <= self.threshold, threshold=self.threshold
                ))
        return reports