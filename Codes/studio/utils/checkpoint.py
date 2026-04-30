# studio/utils/checkpoint.py
import json
import time
from pathlib import Path

class CheckpointManager:
    def __init__(self, checkpoint_dir="./checkpoints"):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
    def save_checkpoint(self, experiment_id, results, name="checkpoint.json"):
        data = {"experiment_id": experiment_id, "results": results, "timestamp": time.time()}
        with open(self.dir / name, "w") as f:
            json.dump(data, f, indent=2)
    def load_checkpoint(self, name="checkpoint.json"):
        path = self.dir / name
        if not path.exists():
            return 0, []
        with open(path) as f:
            d = json.load(f)
        return d.get("experiment_id", 0), d.get("results", [])