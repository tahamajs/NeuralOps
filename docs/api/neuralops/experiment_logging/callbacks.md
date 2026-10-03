Module neuralops.experiment_logging.callbacks
=============================================
HuggingFace Trainer callback for automatic metric logging.

Classes
-------

`LoggingCallback(tracker: neuralops.experiment_logging.tracker.MetricsTracker, logger: logging.Logger)`
:   Automatically captures training/evaluation logs and stores them in a MetricsTracker.
    
    Usage:
        tracker = MetricsTracker("metrics.json")
        trainer.add_callback(LoggingCallback(tracker, logger))

    ### Ancestors (in MRO)

    * transformers.trainer_callback.TrainerCallback

    ### Methods

    `on_evaluate(self, args, state, control, metrics=None, **kwargs) ‑> None`
    :   On evaluate.

    `on_log(self, args, state, control, logs=None, **kwargs) ‑> None`
    :   On log.

    `on_train_end(self, args, state, control, **kwargs) ‑> None`
    :   On train end.