# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tracking Callbacks

Training callbacks that stream metrics, parameters, and loss trajectories to experiment trackers.
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
