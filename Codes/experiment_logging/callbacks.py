"""Callbacks to integrate logging with Hugging Face Trainer."""

from transformers.trainer_callback import TrainerCallback
from .tracker import MetricsTracker
import torch
import time
import logging

class LoggingCallback(TrainerCallback):
    """
    Automatically captures training/evaluation logs and tracks them.

    Usage:
        from experiment_logging import LoggingCallback, MetricsTracker
        tracker = MetricsTracker("metrics.json")
        trainer.add_callback(LoggingCallback(tracker, logger))
    """
    def __init__(self, tracker: MetricsTracker, logger: logging.Logger):
        self.tracker = tracker
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Record metrics with step
            step = state.global_step
            self.tracker.log(logs, step=step)
            # Also log key metrics to the logger
            if "loss" in logs:
                self.logger.info(f"Step {step}: loss={logs['loss']:.4f}")
            if "eval_loss" in logs:
                self.logger.info(f"Eval step {step}: loss={logs['eval_loss']:.4f}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self.logger.info("=== Evaluation ===")
            for key, value in metrics.items():
                self.logger.info(f"  {key}: {value:.4f}")
            # Save to tracker
            self.tracker.log(metrics, step=state.global_step)
            self.tracker._flush()