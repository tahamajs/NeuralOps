import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .callbacks import GradientNormCallback, MemoryCallback, TimeCallback
from .utils import plot_training_curves


def _sanitize_json_value(value: Any) -> Any:
    """Make training logs strict JSON without hiding non-finite metrics."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_json_value(v) for v in value]
    return value


class ResultsManager:
    """Manages per‑run result directories, logging, metrics, and trainer wrapping."""

    def __init__(
        self,
        experiment_name: str,
        base_dir: str = "results",
        config: dict[str, Any] | None = None,
        log_to_file: bool = True,
    ) -> None:
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
        self.history: list[dict[str, Any]] = []

        if config:
            with open(self.run_dir / "config.yaml", "w") as f:
                yaml.dump(config, f, default_flow_style=False)

        self.logger.info(f"Results directory: {self.run_dir}")

    def _setup_logger(self) -> logging.Logger:
        """setup logger."""
        logger = logging.getLogger(f"training.{self.experiment_name}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(ch)
        if self.log_to_file:
            fh = logging.FileHandler(self.log_dir / "run.log")
            fh.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(fh)
        return logger

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics."""
        record = {"step": step} if step is not None else {}
        record.update(metrics)
        record["timestamp"] = datetime.now().isoformat()
        record = _sanitize_json_value(record)
        self.history.append(record)
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")

    def save_plots(self) -> None:
        """Persist plots to disk."""
        if self.history:
            plot_path = self.run_dir / "training_history.png"
            plot_training_curves(self.history, str(plot_path))
            self.logger.info(f"Saved training plot to {plot_path}")

    def save_final_model(self, model, tokenizer=None) -> None:
        """Persist final model to disk."""
        model.save_pretrained(self.final_model_dir)
        if tokenizer:
            tokenizer.save_pretrained(self.final_model_dir)
        self.logger.info(f"Final model saved to {self.final_model_dir}")

    def get_checkpoint_dir(self) -> str:
        """Return the checkpoint dir."""
        return str(self.checkpoint_dir)

    def push_to_hf(self, repo_id: str | None = None) -> None:
        """Push full run results (models, plots, metrics) to HuggingFace Hub."""
        from neuralops.results_saver.backends import huggingface as hf_backend

        cfg = {"hf_options": {"repo_id": repo_id}} if repo_id else self.config
        hf_backend.push(self.experiment_name, self.run_dir, cfg)

    def generate_summary_report(self, final_metrics: dict[str, Any] | None = None) -> None:
        """Generate structured markdown and json summary reports for the run."""

        def _make_serializable(obj):
            if hasattr(obj, "to_dict") and callable(obj.to_dict):
                return obj.to_dict()
            if hasattr(obj, "__dict__"):
                return {
                    k: _make_serializable(v)
                    for k, v in obj.__dict__.items()
                    if not k.startswith("_")
                }
            if isinstance(obj, dict):
                return {str(k): _make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple, set)):
                return [_make_serializable(item) for item in obj]
            if isinstance(obj, (datetime, Path)):
                return str(obj)
            if hasattr(obj, "value"):
                return obj.value
            try:
                json.dumps(obj)
                return obj
            except Exception:
                return str(obj)

        summary = {
            "experiment_name": self.experiment_name,
            "run_dir": str(self.run_dir),
            "total_steps": len(self.history),
            "config": _make_serializable(self.config),
            "final_metrics": _sanitize_json_value(
                _make_serializable(final_metrics or (self.history[-1] if self.history else {}))
            ),
            "timestamp": datetime.now().isoformat(),
        }
        summary_path = self.run_dir / "run_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, allow_nan=False)

        card_path = self.run_dir / "run_summary.md"
        with open(card_path, "w") as f:
            f.write(f"# Run Summary: {self.experiment_name}\n\n")
            f.write(f"- **Timestamp**: {summary['timestamp']}\n")
            f.write(f"- **Total Logging Steps**: {summary['total_steps']}\n")
            f.write(f"- **Run Directory**: `{self.run_dir}`\n\n")
            f.write("## Final Metrics\n```json\n")
            f.write(json.dumps(summary["final_metrics"], indent=2))
            f.write("\n```\n")
        self.logger.info(f"Generated structured run summary report at {card_path}")

    def wrap_trainer(self, trainer) -> Any:
        """Add standard callbacks and metrics logging to a Hugging Face Trainer."""
        try:
            trainer.add_callback(MemoryCallback())
            trainer.add_callback(TimeCallback())
            trainer.add_callback(GradientNormCallback(log_every_n_steps=100))
        except Exception as e:
            self.logger.warning(f"Could not add callbacks: {e}")

        original_log = trainer.log

        def custom_log(logs) -> None:
            """Custom log."""
            original_log(logs)
            self.log_metrics(logs)

        trainer.log = custom_log
        return trainer
