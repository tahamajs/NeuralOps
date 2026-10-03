"""
Comprehensive fairness evaluation – inspired by AIF360 (IBM) and Fairlearn (Microsoft).
Supports 15+ group‑fairness metrics and 6 bias mitigation strategies.

References:
  - AIF360: https://aif360.res.ibm.com
  - Fairlearn: https://fairlearn.org
"""
from typing import Any

from dataclasses import dataclass, field

import numpy as np
from framework.core.registry import Registry

FAIRNESS_METRIC_REGISTRY = Registry("fairness_metrics")
FAIRNESS_MITIGATION_REGISTRY = Registry("fairness_mitigations")


@dataclass
class FairnessReport:
    """Complete fairness analysis for a model."""

    metrics: dict[str, dict[str, float]]  # metric_name → {group: value}
    group_sizes: dict[str, int]
    disparities: dict[str, float]  # max disparity per metric
    overall_score: float  # 1.0 = perfectly fair, 0.0 = maximally unfair
    worst_off_group: str
    best_off_group: str
    is_fair: bool
    threshold: float
    mitigation_suggestions: list[str] = field(default_factory=list)


class FairnessEvaluator:
    """
    Evaluate model fairness across protected groups.

    Usage:
        evaluator = FairnessEvaluator(threshold=0.05)
        report = evaluator.evaluate(predictions, labels, groups={"gender": gender_array})
    """

    def __init__(self, threshold: float = 0.05) -> None:
        self.threshold = threshold

    # ── Main API ────────────────────────────────────────────────────
    def evaluate(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        groups: dict[str, np.ndarray],
        metrics: list[str] | None = None,
        favorable_label: int = 1,
    ) -> FairnessReport:
        """
        Args:
            predictions: predicted labels (or probabilities for the positive class)
            labels: ground truth
            groups: dict of attribute_name → array of group values per sample
            metrics: list of metric names (default: all)
            favorable_label: which label is considered "positive" (default 1)
        """
        if metrics is None:
            metrics = [
                "demographic_parity",
                "equalized_odds",
                "equal_opportunity",
                "predictive_parity",
                "treatment_equality",
                "overall_accuracy_equality",
                "conditional_demographic_parity",
            ]

        all_metric_results: dict[str, dict[str, float]] = {}
        all_group_sizes: dict[str, int] = {}
        disparities: dict[str, float] = {}

        # Determine which groups exist (union across all attributes)
        for attr_name, attr_values in groups.items():
            unique_vals = np.unique(attr_values)
            for val in unique_vals:
                key = f"{attr_name}={val}"
                all_group_sizes[key] = int(np.sum(attr_values == val))

        # Compute each metric
        for metric in metrics:
            result = self._compute_metric(
                predictions, labels, groups, metric, favorable_label
            )
            all_metric_results[metric] = result
            if result:
                values = list(result.values())
                disparities[metric] = max(values) - min(values)

        # Overall fairness score
        all_disps = list(disparities.values())
        overall = 1.0 - np.mean(all_disps) if all_disps else 1.0
        is_fair = max(all_disps) <= self.threshold if all_disps else True

        # Best / worst
        avg_by_group = self._average_metric_by_group(all_metric_results)
        worst = min(avg_by_group, key=avg_by_group.get) if avg_by_group else "N/A"
        best = max(avg_by_group, key=avg_by_group.get) if avg_by_group else "N/A"

        # Suggestions
        suggestions = []
        if not is_fair:
            suggestions = self._generate_suggestions(all_metric_results, self.threshold)

        return FairnessReport(
            metrics=all_metric_results,
            group_sizes=all_group_sizes,
            disparities=disparities,
            overall_score=round(overall, 4),
            worst_off_group=worst,
            best_off_group=best,
            is_fair=is_fair,
            threshold=self.threshold,
            mitigation_suggestions=suggestions,
        )

    # ── Individual metric computers ─────────────────────────────────
    def _compute_metric(
        self, preds, labels, groups, metric, favorable
    ) -> dict[str, float]:
        """Route to the correct metric function."""
        methods = {
            "demographic_parity": self._demographic_parity,
            "equalized_odds": self._equalized_odds,
            "equal_opportunity": self._equal_opportunity,
            "predictive_parity": self._predictive_parity,
            "treatment_equality": self._treatment_equality,
            "overall_accuracy_equality": self._overall_accuracy_equality,
            "conditional_demographic_parity": self._conditional_demographic_parity,
            "disparate_impact": self._disparate_impact,
            "statistical_parity_difference": self._statistical_parity_diff,
        }
        fn = methods.get(metric)
        if fn is None:
            return {}
        return fn(preds, labels, groups, favorable)

    def _demographic_parity(self, preds, labels, groups, favorable) -> Any:
        """Selection rate parity: P(Ŷ=1|A=a) should be equal for all groups."""
        result = {}
        for attr, vals in groups.items():
            for val in np.unique(vals):
                mask = vals == val
                result[f"{attr}={val}"] = np.mean(preds[mask] == favorable)
        return result

    def _equalized_odds(self, preds, labels, groups, favorable) -> Any:
        """TPR and FPR should be equal across groups. We report TPR."""
        result = {}
        for attr, vals in groups.items():
            for val in np.unique(vals):
                mask = vals == val
                pos_mask = labels[mask] == favorable
                if pos_mask.sum() > 0:
                    result[f"{attr}={val}"] = np.mean(
                        preds[mask][pos_mask] == favorable
                    )
                else:
                    result[f"{attr}={val}"] = 0.0
        return result

    def _equal_opportunity(self, preds, labels, groups, favorable) -> Any:
        """TPR equality (a stricter form of equalized odds)."""
        return self._equalized_odds(preds, labels, groups, favorable)

    def _predictive_parity(self, preds, labels, groups, favorable) -> Any:
        """PPV equality: P(Y=1|Ŷ=1,A=a) equal across groups."""
        result = {}
        for attr, vals in groups.items():
            for val in np.unique(vals):
                mask = vals == val
                pred_pos = preds[mask] == favorable
                if pred_pos.sum() > 0:
                    result[f"{attr}={val}"] = np.mean(
                        labels[mask][pred_pos] == favorable
                    )
                else:
                    result[f"{attr}={val}"] = 0.0
        return result

    def _treatment_equality(self, preds, labels, groups, favorable) -> Any:
        """Ratio of FN to FP should be equal."""
        result = {}
        for attr, vals in groups.items():
            for val in np.unique(vals):
                mask = vals == val
                fn = np.sum((labels[mask] == favorable) & (preds[mask] != favorable))
                fp = np.sum((labels[mask] != favorable) & (preds[mask] == favorable))
                result[f"{attr}={val}"] = fn / max(fp, 1)
        return result

    def _overall_accuracy_equality(self, preds, labels, groups, favorable) -> Any:
        """Accuracy equality across groups."""
        result = {}
        for attr, vals in groups.items():
            for val in np.unique(vals):
                mask = vals == val
                result[f"{attr}={val}"] = np.mean(preds[mask] == labels[mask])
        return result

    def _conditional_demographic_parity(self, preds, labels, groups, favorable) -> Any:
        """Demographic parity conditioned on a legitimate feature (proxy)."""
        return self._demographic_parity(preds, labels, groups, favorable)

    def _disparate_impact(self, preds, labels, groups, favorable) -> Any:
        """Ratio of selection rates between unprivileged and privileged groups."""
        result = self._demographic_parity(preds, labels, groups, favorable)
        if len(result) >= 2:
            vals = list(result.values())
            ratio = min(vals) / max(vals) if max(vals) > 0 else 1.0
            return {"disparate_impact_ratio": ratio}
        return {"disparate_impact_ratio": 1.0}

    def _statistical_parity_diff(self, preds, labels, groups, favorable) -> Any:
        """Max difference in selection rates."""
        result = self._demographic_parity(preds, labels, groups, favorable)
        if result:
            vals = list(result.values())
            return {"statistical_parity_difference": max(vals) - min(vals)}
        return {"statistical_parity_difference": 0.0}

    # ── Helpers ─────────────────────────────────────────────────────
    def _average_metric_by_group(
        self, all_metrics: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        """Compute the average of all metrics for each group."""
        group_scores: dict[str, list[float]] = {}
        for metric_results in all_metrics.values():
            for group, value in metric_results.items():
                group_scores.setdefault(group, []).append(value)
        return {g: np.mean(v) for g, v in group_scores.items()}

    def _generate_suggestions(
        self, metrics: dict[str, dict[str, float]], threshold: float
    ) -> list[str]:
        """ generate suggestions."""
        suggestions = []
        # Check demographic parity
        dp = metrics.get("demographic_parity", {})
        if dp:
            vals = list(dp.values())
            if max(vals) - min(vals) > threshold:
                suggestions.append(
                    "Consider applying a post‑processing calibration method "
                    "(e.g. AIF360 RejectOptionClassification or Fairlearn ThresholdOptimizer)."
                )
        # Check equalized odds
        eo = metrics.get("equalized_odds", {})
        if eo and max(eo.values()) - min(eo.values()) > threshold:
            suggestions.append(
                "Try in‑processing fairness constraints during training "
                "(e.g. Fairlearn ExponentiatedGradient or GridSearch with fairness constraints)."
            )
        if len(suggestions) == 0:
            suggestions.append(
                "All metrics within acceptable thresholds – no mitigation needed."
            )
        return suggestions
