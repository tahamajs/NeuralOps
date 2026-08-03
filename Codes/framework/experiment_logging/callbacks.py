"""HuggingFace Trainer callback for automatic metric logging."""

import logging

from transformers.trainer_callback import TrainerCallback

from .tracker import MetricsTracker


class LoggingCallback(TrainerCallback):
    """
    Automatically captures training/evaluation logs and stores them in a MetricsTracker.

    Usage:
        tracker = MetricsTracker("metrics.json")
        trainer.add_callback(LoggingCallback(tracker, logger))
    """

    def __init__(self, tracker: MetricsTracker, logger: logging.Logger) -> None:
        self.tracker = tracker
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        """On log."""
        if logs:
            step = state.global_step
            self.tracker.log(logs, step=step)
            # Log key metrics to the console/file logger
            loss = logs.get("loss", None)
            eval_loss = logs.get("eval_loss", None)
            lr = logs.get("learning_rate", None)
            msg = f"Step {step}: "
            if loss is not None:
                msg += f"loss={loss:.4f} "
            if eval_loss is not None:
                msg += f"eval_loss={eval_loss:.4f} "
            if lr is not None:
                msg += f"lr={lr:.2e}"
            self.logger.info(msg)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs) -> None:
        """On evaluate."""
        if metrics:
            self.logger.info("=== Evaluation ===")
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.logger.info(f"  {k}: {v:.4f}")
                else:
                    self.logger.info(f"  {k}: {v}")
            self.tracker.log(metrics, step=state.global_step)
            self.tracker.flush()

    def on_train_end(self, args, state, control, **kwargs) -> None:
        """On train end."""
        self.tracker.save()
        self.logger.info("Training finished. Metrics saved.")
