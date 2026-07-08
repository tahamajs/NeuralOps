"""Training‑related callbacks: logging, metrics, early stopping, checkpoint, LR."""

import json
import logging
import os

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class LoggingCallback(TrainerCallback):
    """Logs training events to a Python logger."""

    def __init__(self, metric_tracker=None, logger_instance=None):
        self.tracker = metric_tracker
        self.log = logger_instance or logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = logs.get("step", state.global_step)
            loss = logs.get("loss", "N/A")
            lr = logs.get("learning_rate", "N/A")
            self.log.info(f"Step {step}: loss={loss}, lr={lr}")
            if self.tracker:
                self.tracker.update(logs)


class MetricsTrackerCallback(TrainerCallback):
    """Collects and saves metrics to a JSON file."""

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.metrics = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.metrics.append(logs)

    def on_train_end(self, args, state, control, **kwargs):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, "w") as f:
            json.dump(self.metrics, f, indent=2)


class ProgressCallback(TrainerCallback):
    """Simple progress logging."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0:
            logger.info(f"Progress: {state.global_step}/{state.max_steps} steps")


class EarlyStoppingCallback(TrainerCallback):
    """Stops training if a metric doesn't improve for `patience` evaluations."""

    def __init__(
        self, metric: str = "eval_loss", patience: int = 3, min_delta: float = 0.0
    ):
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = None
        self.wait = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        current = metrics.get(self.metric)
        if current is None:
            return
        if self.best_metric is None or current < self.best_metric - self.min_delta:
            self.best_metric = current
            self.wait = 0
        else:
            self.wait += 1
            logger.info(f"EarlyStopping: patience {self.wait}/{self.patience}")
        if self.wait >= self.patience:
            logger.info("EarlyStopping: stopping training")
            control.should_training_stop = True


class CheckpointCallback(TrainerCallback):
    """Saves model checkpoints during training."""

    def __init__(self, output_dir: str, save_every_steps: int = 500):
        self.output_dir = output_dir
        self.save_every_steps = save_every_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.save_every_steps == 0:
            model = kwargs.get("model")
            if model is not None:
                path = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
                model.save_pretrained(path)
                logger.info(f"Checkpoint saved to {path}")


class LRSchedulerCallback(TrainerCallback):
    """Logs learning rate changes."""

    def on_step_end(self, args, state, control, **kwargs):
        if "lr_scheduler" in kwargs:
            lr = kwargs["lr_scheduler"].get_last_lr()[0]
            logger.debug(f"Learning rate: {lr:.2e}")


class MetricsLoggerCallback(TrainerCallback):
    """Persists metrics using a ResultsSaver."""

    def __init__(self, saver):
        self.saver = saver  # Expects an object with save_metrics(dict, step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and self.saver:
            self.saver.save_metrics(logs, state.global_step)
