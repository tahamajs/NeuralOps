Module neuralops.evaluation.label_quality
=========================================
Label quality detection – inspired by Cleanlab, the industry standard for
data‑centric AI.  Finds label errors, outliers, and near‑duplicates.

Reference: https://github.com/cleanlab/cleanlab

Classes
-------

`LabelQualityDetector(threshold: float = 0.1, seed: int = 42)`
:   Detects label errors, outliers, and near‑duplicates using confident learning.
    
    Usage:
        detector = LabelQualityDetector()
        report = detector.analyze(labels=y_true, pred_probs=model_output_probs)
    
    No ground‑truth validation set is required – the method works entirely from
    model predictions and given labels.

    ### Methods

    `analyze(self, labels: numpy.ndarray, pred_probs: numpy.ndarray, class_names: list[str] | None = None) ‑> neuralops.evaluation.label_quality.LabelQualityReport`
    :   Run full label quality analysis.
        
        Args:
            labels: integer labels, shape (N,)
            pred_probs: model predicted probabilities, shape (N, K)
            class_names: optional list of class name strings

    `summary(self, report: neuralops.evaluation.label_quality.LabelQualityReport) ‑> dict[str, typing.Any]`
    :   Summary.

`LabelQualityReport(n_samples: int, n_label_issues: int, issue_rate: float, label_issue_indices: numpy.ndarray, label_quality_scores: numpy.ndarray, suggested_labels: numpy.ndarray | None = None, per_class_issues: dict[int, int] = <factory>, outlier_indices: numpy.ndarray = <factory>, near_duplicate_pairs: list[tuple[int, int]] = <factory>, overall_quality_score: float = 0.0)`
:   Complete label quality analysis for a dataset.

    ### Instance variables

    `issue_rate: float`
    :   The type of the None singleton.

    `label_issue_indices: numpy.ndarray`
    :   The type of the None singleton.

    `label_quality_scores: numpy.ndarray`
    :   The type of the None singleton.

    `n_label_issues: int`
    :   The type of the None singleton.

    `n_samples: int`
    :   The type of the None singleton.

    `near_duplicate_pairs: list[tuple[int, int]]`
    :   The type of the None singleton.

    `outlier_indices: numpy.ndarray`
    :   The type of the None singleton.

    `overall_quality_score: float`
    :   The type of the None singleton.

    `per_class_issues: dict[int, int]`
    :   The type of the None singleton.

    `suggested_labels: numpy.ndarray | None`
    :   The type of the None singleton.