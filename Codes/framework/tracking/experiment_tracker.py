import json
import os
from datetime import datetime

class ExperimentTracker:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.log_dir = self.config.get("log_dir", "./experiments/logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        self.metrics = []

    def log(self, metrics: Dict[str, Any], step: int = None, prefix: str = ""):
        entry = {"timestamp": datetime.now().isoformat(), "step": step, **metrics}
        self.metrics.append(entry)

    def finalize(self, result: Dict[str, Any] = None):
        with open(self.log_file, 'w') as f:
            json.dump({"history": self.metrics, "result": result}, f, indent=2)