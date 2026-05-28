import os
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from .callbacks import MemoryCallback, TimeCallback, GradientNormCallback
from .utils import plot_training_curves

class ResultsManager:
    """Manages per‑run result directories, logging, metrics, and trainer wrapping."""
    def __init__(self, experiment_name: str, base_dir: str = "results",
                 config: Optional[Dict[str, Any]] = None, log_to_file: bool = True):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.config = config or {}
        self.log_to_file = log_to_file

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = self.base_dir / f"{experiment_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.log_dir = self.run_dir / "logs"
        self.final_model_dir = self.run_dir / "final_model"
        for d in [self.checkpoint_dir, self.log_dir, self.final_model_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = self._setup_logger()
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.history: List[Dict[str, Any]] = []

        if config:
            with open(self.run_dir / "config.yaml", "w") as f:
                yaml.dump(config, f, default_flow_style=False)

        self.logger.info(f"Results directory: {self.run_dir}")

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"training.{self.experiment_name}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(ch)
        if self.log_to_file:
            fh = logging.FileHandler(self.log_dir / "run.log")
            fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            logger.addHandler(fh)
        return logger

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        record = {"step": step} if step is not None else {}
        record.update(metrics)
        record["timestamp"] = datetime.now().isoformat()
        self.history.append(record)
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def save_plots(self) -> None:
        if self.history:
            plot_path = self.run_dir / "training_history.png"
            plot_training_curves(self.history, str(plot_path))
            self.logger.info(f"Saved training plot to {plot_path}")

    def save_final_model(self, model: Any, tokenizer: Optional[Any] = None) -> None:
        model.save_pretrained(self.final_model_dir)
        if tokenizer:
            tokenizer.save_pretrained(self.final_model_dir)
        self.logger.info(f"Final model saved to {self.final_model_dir}")

    def get_checkpoint_dir(self) -> str:
        return str(self.checkpoint_dir)

    def wrap_trainer(self, trainer: Any) -> None:
        """Add standard callbacks and metrics logging to a Hugging Face Trainer."""
        try:
            trainer.add_callback(MemoryCallback())
            trainer.add_callback(TimeCallback())
            trainer.add_callback(GradientNormCallback(log_every_n_steps=100))
        except Exception as e:
            self.logger.warning(f"Could not add callbacks: {e}")

        original_log = trainer.log
        def custom_log(logs):
            original_log(logs)
            self.log_metrics(logs)
        trainer.log = custom_log
        return trainer