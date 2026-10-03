Module neuralops.evaluation.fairness
====================================
Comprehensive fairness evaluation – inspired by AIF360 (IBM) and Fairlearn (Microsoft).
Supports 15+ group‑fairness metrics and 6 bias mitigation strategies.

References:
  - AIF360: https://aif360.res.ibm.com
  - Fairlearn: https://fairlearn.org

Classes
-------

`FairnessEvaluator(threshold: float = 0.05)`
:   Evaluate model fairness across protected groups.
    
    Usage:
        evaluator = FairnessEvaluator(threshold=0.05)
        report = evaluator.evaluate(predictions, labels, groups={"gender": gender_array})

    ### Methods

    `evaluate(self, predictions: numpy.ndarray, labels: numpy.ndarray, groups: dict[str, numpy.ndarray], metrics: list[str] | None = None, favorable_label: int = 1) ‑> neuralops.evaluation.fairness.FairnessReport`
    :   Args:
            predictions: predicted labels (or probabilities for the positive class)
            labels: ground truth
            groups: dict of attribute_name → array of group values per sample
            metrics: list of metric names (default: all)
            favorable_label: which label is considered "positive" (default 1)

`FairnessReport(metrics: dict[str, dict[str, float]], group_sizes: dict[str, int], disparities: dict[str, float], overall_score: float, worst_off_group: str, best_off_group: str, is_fair: bool, threshold: float, mitigation_suggestions: list[str] = <factory>)`
:   Complete fairness analysis for a model.

    ### Instance variables

    `best_off_group: str`
    :   The type of the None singleton.

    `disparities: dict[str, float]`
    :   The type of the None singleton.

    `group_sizes: dict[str, int]`
    :   The type of the None singleton.

    `is_fair: bool`
    :   The type of the None singleton.

    `metrics: dict[str, dict[str, float]]`
    :   The type of the None singleton.

    `mitigation_suggestions: list[str]`
    :   The type of the None singleton.

    `overall_score: float`
    :   The type of the None singleton.

    `threshold: float`
    :   The type of the None singleton.

    `worst_off_group: str`
    :   The type of the None singleton.