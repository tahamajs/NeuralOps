Module neuralops.tracking.callbacks
===================================
HuggingFace Trainer callback that logs metrics and parameters to an ExperimentTracker.

Classes
-------

`TrackerCallback(tracker: neuralops.tracking.tracker.ExperimentTracker)`
:   Automatically log training/evaluation metrics to the tracker.

    ### Ancestors (in MRO)

    * transformers.trainer_callback.TrainerCallback

    ### Methods

    `on_evaluate(self, args, state, control, logs=None, **kwargs) ‑> None`
    :   On evaluate.

    `on_log(self, args, state, control, logs=None, **kwargs) ‑> None`
    :   On log.

    `on_train_end(self, args, state, control, **kwargs) ‑> None`
    :   On train end.