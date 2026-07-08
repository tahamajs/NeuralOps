import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ResultsSaver:
    def __init__(
        self, run_id: str, base_dir: Path, config: Optional[Dict[str, Any]] = None
    ):
        self.run_id = run_id
        self.base_dir = base_dir / str(run_id)
        self.config = config or {}
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_config(self, config: Dict[str, Any]):
        config_path = self.base_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved config to {config_path}")

    def save_history(self, history: Dict[str, Any]):
        history_path = self.base_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Saved history to {history_path}")

    def save_final_metrics(self, metrics: Dict[str, Any]):
        metrics_path = self.base_dir / "final_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")

    def generate_plots(self, history: Dict[str, Any]):
        plot_dir = self.base_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        # Placeholder for plot generation logic
        logger.info(f"Plots would be generated in {plot_dir}")

    def should_save_model(self, metrics: Dict[str, Any]) -> bool:
        if not self.config.get("save_as_version", True):
            return False
        if "eval_accuracy" in metrics:
            return metrics["eval_accuracy"] > 0.7
        return True

    def handle_storage_backend(self):
        backend = self.config.get("storage_backend", "local")
        if backend != "local":
            logger.info(f"Storage backend {backend} not implemented")
