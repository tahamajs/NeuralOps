"""
HuggingFace Trainer callback that logs metrics and parameters to an ExperimentTracker.
"""

from transformers import TrainerCallback

from .tracker import ExperimentTracker


class TrackerCallback(TrainerCallback):
    """Automatically log training/evaluation metrics to the tracker."""

    def __init__(self, tracker: ExperimentTracker) -> None:
        self.tracker = tracker

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        """On log."""
        if logs:
            step = state.global_step
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tracker.log_metric(k, v, step=step)

    def on_evaluate(self, args, state, control, logs=None, **kwargs) -> None:
        """On evaluate."""
        if logs:
            step = state.global_step
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tracker.log_metric(f"eval_{k}", v, step=step)

    def on_train_end(self, args, state, control, **kwargs) -> None:
        """On train end."""
        self.tracker.finalize()
