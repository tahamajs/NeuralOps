"""Simple metrics tracker for experiments."""
import json
import os
from typing import Dict, List, Any, Optional

class MetricsTracker:
    """Track scalar metrics across steps/epochs."""
    def __init__(self, save_path: Optional[str] = None):
        self.metrics: List[Dict[str, Any]] = []
        self.save_path = save_path

    def log(self, metrics: Dict[str, float], step: int = None):
        entry = metrics.copy()
        if step is not None:
            entry["step"] = step
        self.metrics.append(entry)
        if self.save_path:
            self._flush()

    def _flush(self):
        with open(self.save_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2)
