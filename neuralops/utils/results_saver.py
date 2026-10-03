"""
Unified ResultsSaver re-exported under framework.utils.results_saver
Integrates persistence, structured markdown/json report generation, Django API compatibility,
and Hugging Face Hub uploads.
"""

import logging
from pathlib import Path
from typing import Any

from neuralops.results_saver.backends import huggingface as hf_backend
from neuralops.results_saver.saver import ResultsSaver as PrimaryResultsSaver

logger = logging.getLogger(__name__)


class ResultsSaver(PrimaryResultsSaver):
    """
    Enhanced ResultsSaver compatible with Django API, experiment runners, and CLI scripts.
    Supports structured report generation and Hugging Face Hub uploads.
    """

    def __init__(
        self, run_id: str, base_dir: Path, config: dict[str, Any] | None = None
    ) -> None:
        super().__init__(run_id=run_id, base_dir=base_dir, config=config)
        self.config = config or {}
        self.legacy_base_dir = Path(base_dir) / str(run_id)
        self.legacy_base_dir.mkdir(parents=True, exist_ok=True)

    def save_config(self, config: dict[str, Any]) -> None:
        """Persist config to disk in structured format."""
        self.config = config
        self._save_config(config)
        self._write_json("../config.json", config)
        logger.info(f"Saved structured config to {self.run_dir}")

    def save_history(self, history: dict[str, Any]) -> None:
        """Persist training history to disk."""
        self._save_history(history)
        self._write_json("../history.json", history)
        logger.info(f"Saved training history to {self.run_dir}")

    def save_final_metrics(self, metrics: dict[str, Any]) -> None:
        """Persist final metrics and generate run report."""
        self._save_final_metrics(metrics)
        self._write_json("../final_metrics.json", metrics)
        self._generate_run_report(metrics)
        logger.info(f"Saved final metrics and generated run report to {self.run_dir}")

    def generate_plots(self, history: dict[str, Any]) -> None:
        """Generate plots."""
        self._generate_plots(history, {})

    def should_save_model(self, metrics: dict[str, Any]) -> bool:
        """Evaluate whether heavy model artifacts should be saved."""
        return self.should_save_heavy(metrics)

    def handle_storage_backend(self) -> None:
        """Execute configured storage backends including HuggingFace Hub."""
        self.push_all()

    def push_to_hf(self, repo_id: str | None = None) -> None:
        """Upload complete run artifacts to HuggingFace Hub for resource management."""
        cfg = {"hf_options": {"repo_id": repo_id}} if repo_id else self.cfg
        hf_backend.push(self.run_id, self.run_dir, cfg)

    def _generate_run_report(self, metrics: dict[str, Any]) -> None:
        """Generate structured run summary report."""
        report = {
            "run_id": self.run_id,
            "status": "COMPLETED",
            "metrics": metrics,
            "config": self.config
            or (self.cfg.to_dict() if hasattr(self.cfg, "to_dict") else str(self.cfg)),
        }
        self._write_json("artifacts/run_report.json", report)
        self._save_model_card(self.config or {}, metrics)
