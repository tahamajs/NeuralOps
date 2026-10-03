Module neuralops.evaluation
===========================
Evaluation module for model performance, fairness, robustness, label quality, and A/B testing.

Sub-modules
-----------
* neuralops.evaluation.ab_testing
* neuralops.evaluation.eval_all_models_datasets
* neuralops.evaluation.eval_perplexity
* neuralops.evaluation.fairness
* neuralops.evaluation.full_mnli_validation
* neuralops.evaluation.label_quality
* neuralops.evaluation.metrics
* neuralops.evaluation.retraining
* neuralops.evaluation.run_baseline_comparison
* neuralops.evaluation.run_lm_eval_harness
* neuralops.evaluation.run_targeted_b2_recovery

Functions
---------

`get_task_metrics(task_name: str) ‑> Any`
:   Return a compute_metrics function for a GLUE task.

Classes
-------

`ABTestResult(metric: str, mean_a: float, mean_b: float, difference: float, p_value: float, significant: bool, better_model: str, confidence_interval: tuple[float, float])`
:   ABTestResult(metric: str, mean_a: float, mean_b: float, difference: float, p_value: float, significant: bool, better_model: str, confidence_interval: tuple[float, float])

    ### Instance variables

    `better_model: str`
    :   The type of the None singleton.

    `confidence_interval: tuple[float, float]`
    :   The type of the None singleton.

    `difference: float`
    :   The type of the None singleton.

    `mean_a: float`
    :   The type of the None singleton.

    `mean_b: float`
    :   The type of the None singleton.

    `metric: str`
    :   The type of the None singleton.

    `p_value: float`
    :   The type of the None singleton.

    `significant: bool`
    :   The type of the None singleton.

`ABTester(alpha: float = 0.05)`
:   

    ### Methods

    `compare(self, results_a: dict[str, list[float]], results_b: dict[str, list[float]]) ‑> list[neuralops.evaluation.ab_testing.ABTestResult]`
    :   Compare.

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

`RetrainDecision(should_retrain: bool, reason: str, trigger_type: neuralops.evaluation.retraining.TriggerType | None = None, urgency: str = 'low', details: dict[str, typing.Any] = <factory>)`
:   Output of the retraining decision engine.

    ### Instance variables

    `details: dict[str, typing.Any]`
    :   The type of the None singleton.

    `reason: str`
    :   The type of the None singleton.

    `should_retrain: bool`
    :   The type of the None singleton.

    `trigger_type: neuralops.evaluation.retraining.TriggerType | None`
    :   The type of the None singleton.

    `urgency: str`
    :   The type of the None singleton.

`RetrainPolicy(trigger_types: list[neuralops.evaluation.retraining.TriggerType] = <factory>, drift_threshold: float = 0.3, drift_methods: list[str] = <factory>, min_accuracy: float = 0.85, max_accuracy_degradation: float = 0.05, schedule_cron: str | None = None, max_days_since_last_train: int = 30, min_new_samples: int = 1000, max_new_samples: int = 100000, require_better_model: bool = True, min_improvement: float = 0.01, max_retrains_per_day: int = 3, validation_split: float = 0.2, notify_on_trigger: bool = True, notify_on_complete: bool = True)`
:   Configuration for when and how to retrain.

    ### Instance variables

    `drift_methods: list[str]`
    :   The type of the None singleton.

    `drift_threshold: float`
    :   The type of the None singleton.

    `max_accuracy_degradation: float`
    :   The type of the None singleton.

    `max_days_since_last_train: int`
    :   The type of the None singleton.

    `max_new_samples: int`
    :   The type of the None singleton.

    `max_retrains_per_day: int`
    :   The type of the None singleton.

    `min_accuracy: float`
    :   The type of the None singleton.

    `min_improvement: float`
    :   The type of the None singleton.

    `min_new_samples: int`
    :   The type of the None singleton.

    `notify_on_complete: bool`
    :   The type of the None singleton.

    `notify_on_trigger: bool`
    :   The type of the None singleton.

    `require_better_model: bool`
    :   The type of the None singleton.

    `schedule_cron: str | None`
    :   The type of the None singleton.

    `trigger_types: list[neuralops.evaluation.retraining.TriggerType]`
    :   The type of the None singleton.

    `validation_split: float`
    :   The type of the None singleton.

`RetrainingOrchestrator(policy: neuralops.evaluation.retraining.RetrainPolicy = None)`
:   Central decision engine for automated retraining.
    
    Usage:
        orch = RetrainingOrchestrator(RetrainPolicy())
        decision = orch.evaluate(
            current_metrics={"accuracy": 0.82},
            reference_metrics={"accuracy": 0.88},
            drift_report=drift_report,
            new_data_count=5000,
            days_since_last_train=35,
        )
        if decision.should_retrain:
            orch.trigger_retraining(run_id="model-v3", training_fn=my_train_func)

    ### Methods

    `evaluate(self, current_metrics: dict[str, float] | None = None, reference_metrics: dict[str, float] | None = None, drift_report: Any | None = None, new_data_count: int = 0, days_since_last_train: int = 0, manual_request: bool = False) ‑> neuralops.evaluation.retraining.RetrainDecision`
    :   Evaluate all configured triggers and return a retraining decision.

    `trigger_retraining(self, run_id: str, training_fn: <class 'collections.abc.Callable'>, decision: neuralops.evaluation.retraining.RetrainDecision, **training_kwargs) ‑> dict[str, typing.Any]`
    :   Execute retraining and validate the new model.

`TriggerType(*values)`
:   Create a collection of name/value pairs.
    
    Example enumeration:
    
    >>> class Color(Enum):
    ...     RED = 1
    ...     BLUE = 2
    ...     GREEN = 3
    
    Access them by:
    
    - attribute access:
    
      >>> Color.RED
      <Color.RED: 1>
    
    - value lookup:
    
      >>> Color(1)
      <Color.RED: 1>
    
    - name lookup:
    
      >>> Color['RED']
      <Color.RED: 1>
    
    Enumerations can be iterated over, and know how many members they have:
    
    >>> len(Color)
    3
    
    >>> list(Color)
    [<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>]
    
    Methods can be added to enumerations, and members can have their own
    attributes -- see the documentation for details.

    ### Ancestors (in MRO)

    * enum.Enum

    ### Class variables

    `DATA_VOLUME`
    :   The type of the None singleton.

    `DRIFT`
    :   The type of the None singleton.

    `MANUAL`
    :   The type of the None singleton.

    `PERFORMANCE`
    :   The type of the None singleton.

    `SCHEDULE`
    :   The type of the None singleton.