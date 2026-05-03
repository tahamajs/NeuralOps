"""
Callbacks that integrate logging with Hugging Face Trainer,
automatically saving training curves to disk at the end of training.
"""

import os
import logging
from pathlib import Path
from transformers.trainer_callback import TrainerCallback
from .metrics import MetricsTracker
from .plotting import plot_training_history


class LoggingCallback(TrainerCallback):
    """
    Captures training/evaluation metrics and saves plots at the end.

    Usage:
        from framework.analysis import LoggingCallback, MetricsTracker
        tracker = MetricsTracker("results/experiment1/metrics.json")
        trainer.add_callback(LoggingCallback(tracker, logger, plots_dir="results/experiment1/plots"))
    """

    def __init__(self, tracker: MetricsTracker, logger: logging.Logger,
                 plots_dir: str = None, save_plots: bool = True):
        self.tracker = tracker
        self.logger = logger
        self.plots_dir = Path(plots_dir) if plots_dir else None
        self.save_plots = save_plots
        self.train_loss = []
        self.eval_loss = []
        self.eval_steps = []
        self.train_steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            self.tracker.log(logs, step=step)
            if "loss" in logs and "eval_loss" not in logs:
                self.logger.info(f"Step {step}: loss={logs['loss']:.4f}")
                self.train_loss.append(logs["loss"])
                self.train_steps.append(step)
            if "eval_loss" in logs:
                self.logger.info(f"Eval step {step}: eval_loss={logs['eval_loss']:.4f}")
                self.eval_loss.append(logs["eval_loss"])
                self.eval_steps.append(step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self.logger.info("=== Evaluation ===")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"  {key}: {value:.4f}")
                else:
                    self.logger.info(f"  {key}: {value}")
            self.tracker.log(metrics, step=state.global_step)
            self.tracker._flush()
            if self.eval_steps and state.global_step not in self.eval_steps:
                # sometimes eval_loss might not be in logs but we want to record eval at this step
                pass

    def on_train_end(self, args, state, control, **kwargs):
        self.tracker.save()
        self.logger.info("Training finished. Metrics saved.")
        if self.save_plots and self.plots_dir:
            self.plots_dir.mkdir(parents=True, exist_ok=True)
            # Save training history plot
            history = {
                "train_steps": self.train_steps,
                "train_loss": self.train_loss,
                "eval_steps": self.eval_steps,
                "eval_loss": self.eval_loss,
            }
            plot_training_history(history, save_path=str(self.plots_dir / "training_history.png"))
            self.logger.info(f"Training plots saved to {self.plots_dir}")