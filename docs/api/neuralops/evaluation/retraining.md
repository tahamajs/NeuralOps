Module neuralops.evaluation.retraining
======================================
Automated retraining pipeline – triggers retraining based on drift, performance
degradation, time schedule, or data volume thresholds.

Inspired by: Vertex AI Model Monitoring, Azure ML pipelines, ClearML, and AutoDrift.

Classes
-------

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