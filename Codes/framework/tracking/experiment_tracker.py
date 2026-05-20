import json
import os
from datetime import datetime
from enum import Enum
from typing import Any, Dict


class TrackingBackend(Enum):
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"
    MLFLOW = "mlflow"
    LOCAL = "local"


class ExperimentTracker:
    """Minimal local experiment tracker with optional backend-friendly API."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.log_dir = self.config.get("log_dir", "./experiments/logs")
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(self.log_dir, f"experiment_{timestamp}.json")
        self.metrics = []

    def log(self, metrics: Dict[str, Any], step: int = None, prefix: str = ""):
        payload = {f"{prefix}{k}" if prefix else k: v for k, v in metrics.items()}
        entry = {"timestamp": datetime.now().isoformat(), "step": step, **payload}
        self.metrics.append(entry)

    def finalize(self, result: Dict[str, Any] = None):
        with open(self.log_file, 'w') as f:
            json.dump({"history": self.metrics, "result": result or {}}, f, indent=2)
