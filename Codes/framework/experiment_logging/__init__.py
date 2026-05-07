""" experiment_logging – minimal standalone package for GLUE finetuning """

import logging
import os
from transformers import TrainerCallback

class MetricsTracker:
    def __init__(self, save_path=None):
        self.save_path = save_path
        self.records = []
    def log(self, metrics, step=None):
        entry = dict(metrics)
        if step is not None: entry["step"] = step
        self.records.append(entry)
    def save(self, path=None):
        p = path or self.save_path
        if p:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            import json
            with open(p, 'w') as f: json.dump(self.records, f, indent=2)
    def flush(self):
        if self.save_path: self.save(self.save_path)

class LoggingCallback(TrainerCallback):
    """Trainer callback that logs metrics via MetricsTracker."""
    def __init__(self, tracker: MetricsTracker, logger: logging.Logger):
        self.tracker = tracker
        self.logger = logger

    def on_train_begin(self, args, state, control, **kwargs):
        self.logger.info("Training started.")

    def on_step_end(self, args, state, control, **kwargs):
        pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            self.tracker.log(logs, step=step)
            loss = logs.get("loss", None)
            eval_loss = logs.get("eval_loss", None)
            lr = logs.get("learning_rate", None)
            msg = f"Step {step}: "
            if loss is not None: msg += f"loss={loss:.4f} "
            if eval_loss is not None: msg += f"eval_loss={eval_loss:.4f} "
            if lr is not None: msg += f"lr={lr:.2e}"
            self.logger.info(msg)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self.logger.info("=== Evaluation ===")
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.logger.info(f"  {k}: {v:.4f}")
                else:
                    self.logger.info(f"  {k}: {v}")
            self.tracker.log(metrics, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        self.tracker.flush()
        self.logger.info("Training completed. Metrics saved.")

def setup_logging(log_dir: str = "./logs", experiment_name: str = "experiment") -> logging.Logger:
    logger = logging.getLogger("GLUE")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, f"{experiment_name}.log"))
    ch = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s', datefmt='%H:%M:%S')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
