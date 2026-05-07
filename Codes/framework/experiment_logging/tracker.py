# framework/experiment_logging/tracker.py
import json
import os
from typing import List, Dict, Any, Optional

class MetricsTracker:
    """Simple in‑memory metrics tracker. No circular imports."""

    def __init__(self, save_path: Optional[str] = None):
        self.save_path = save_path
        self.records: List[Dict[str, Any]] = []

    def log(self, metrics: Dict[str, Any], step: int = None):
        entry = dict(metrics)
        if step is not None:
            entry["step"] = step
        self.records.append(entry)

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.records)

    def save(self, path: Optional[str] = None):
        p = path or self.save_path
        if p:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, 'w') as f:
                json.dump(self.records, f, indent=2)

    def flush(self):
        if self.save_path:
            self.save(self.save_path)